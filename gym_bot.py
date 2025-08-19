#!/usr/bin/env python3
import logging
import json
import os
import sqlite3
import asyncio
from datetime import datetime, timedelta
from io import BytesIO

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import TELEGRAM_BOT_TOKEN

# ----------------- Logging Setup -----------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('gym_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------- Conversation States -----------------
(
    MAIN_MENU,
    EXERCISE_GROUP,
    EXERCISE_SELECTION,
    INPUT_WEIGHT,
    INPUT_REPS,
    INPUT_SETS,
    VIEW_PROGRESS,
) = range(7)

# ----------------- Exercise Categories -----------------
EXERCISES = {
    "push": [
        "Bench Press", "Overhead Press", "Push-ups", "Dips", "Incline Bench",
        "Arnold Press", "Chest Fly", "Close-grip Bench", "Diamond Push-ups", "Pike Push-ups"
    ],
    "pull": [
        "Deadlift", "Pull-ups", "Barbell Row", "Dumbbell Row", "Chin-ups",
        "Face Pulls", "Lat Pulldown", "Inverted Row", "Seated Cable Row", "Shrugs"
    ],
    "legs": [
        "Squat", "Lunges", "Leg Press", "Romanian Deadlift", "Step-ups",
        "Hip Thrust", "Leg Extension", "Leg Curl", "Calf Raise", "Bulgarian Split Squat"
    ],
    "core": [
        "Plank", "Hanging Leg Raise", "Russian Twist", "Ab Wheel Rollout", "Sit-ups",
        "Bicycle Crunches", "Mountain Climbers", "Flutter Kicks", "V-Ups", "Side Plank"
    ],
    "bodyweight": [
        "Push-ups", "Pull-ups", "Squats", "Dips", "Lunges",
        "Plank", "Burpees", "Mountain Climbers", "Jump Squats", "Handstand Hold"
    ]
}

# Volume thresholds for exertion categorization
VOLUME_THRESHOLDS = {
    "light": (0, 99),
    "moderate": (100, 199),
    "heavy": (200, 349),
    "very_heavy": (350, 499),
    "extreme": (500, float('inf'))
}

VOLUME_CATEGORIES = {
    "light": {"emoji": "🟢", "description": "Light session - recovery or warm-up"},
    "moderate": {"emoji": "🟡", "description": "Moderate workout - good maintenance"},
    "heavy": {"emoji": "🟠", "description": "Heavy session - solid training"},
    "very_heavy": {"emoji": "🔴", "description": "Very heavy - intense workout"},
    "extreme": {"emoji": "🔥", "description": "Extreme volume - beast mode!"}
}

# ----------------- Database Manager -----------------
class GymDatabase:
    """Handles all database operations for the gym tracker."""
    
    def __init__(self, db_file="gym_tracker.db"):
        self.db_file = db_file
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with proper schema."""
        with sqlite3.connect(self.db_file) as conn:
            # Create table - if it exists, this won't overwrite it
            conn.execute('''
                CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    exercise TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0,
                    reps INTEGER NOT NULL DEFAULT 1,
                    sets INTEGER NOT NULL DEFAULT 1,
                    exertion REAL NOT NULL
                )
            ''')
            
            # Check existing columns and add created_at if missing
            cursor = conn.execute("PRAGMA table_info(workouts)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'created_at' not in columns:
                try:
                    conn.execute('ALTER TABLE workouts ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
                    logger.info("Added created_at column to existing workouts table")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not add created_at column: {e}")
            
            # Create indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_date ON workouts(user_id, date)')
            
            # Only create created_at index if the column exists
            cursor = conn.execute("PRAGMA table_info(workouts)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'created_at' in columns:
                conn.execute('CREATE INDEX IF NOT EXISTS idx_user_created ON workouts(user_id, created_at)')
            
            conn.commit()
        logger.info("Database initialized successfully")
    
    def _calculate_exertion(self, weight, reps, sets):
        """
        Calculate exertion score using the formula: (weight * reps * sets) / 10
        For bodyweight exercises, use default weight estimation.
        """
        # Handle bodyweight exercises with estimated weight
        if weight == 0:
            weight = 70  # Average bodyweight estimation for bodyweight exercises
        
        base_exertion = (weight * reps * sets) / 10
        return round(base_exertion, 2)
    
    def log_workout(self, user_id, exercise, weight, reps, sets):
        """Log a workout to the database."""
        try:
            exertion = self._calculate_exertion(weight, reps, sets)
            date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_file) as conn:
                conn.execute('''
                    INSERT INTO workouts (user_id, date, exercise, weight, reps, sets, exertion)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, date, exercise, weight, reps, sets, exertion))
                conn.commit()
            
            logger.info(f"Logged workout for user {user_id}: {exercise} {weight}kg x {reps} x {sets}")
            return exertion
            
        except Exception as e:
            logger.error(f"Failed to log workout: {e}")
            return None
    
    def get_daily_stats(self, user_id, date=None):
        """Get daily workout statistics."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as exercise_count,
                    SUM(exertion) as total_exertion,
                    AVG(exertion) as avg_exertion
                FROM workouts 
                WHERE user_id = ? AND date = ?
            ''', (user_id, date))
            
            result = cursor.fetchone()
            return {
                'exercise_count': result[0] or 0,
                'total_exertion': result[1] or 0,
                'avg_exertion': round(result[2] or 0, 2),
                'date': date
            }
    
    def get_period_stats(self, user_id, days=7):
        """Get statistics for a period (default 7 days)."""
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute('''
                SELECT 
                    date,
                    COUNT(*) as exercise_count,
                    SUM(exertion) as daily_exertion
                FROM workouts 
                WHERE user_id = ? AND date BETWEEN ? AND ?
                GROUP BY date
                ORDER BY date DESC
            ''', (user_id, start_date, end_date))
            
            daily_stats = cursor.fetchall()
            
            # Get total stats for the period
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_exercises,
                    SUM(exertion) as total_exertion,
                    AVG(exertion) as avg_exertion
                FROM workouts 
                WHERE user_id = ? AND date BETWEEN ? AND ?
            ''', (user_id, start_date, end_date))
            
            total_stats = cursor.fetchone()
            
            return {
                'daily_breakdown': daily_stats,
                'total_exercises': total_stats[0] or 0,
                'total_exertion': total_stats[1] or 0,
                'avg_exertion': round(total_stats[2] or 0, 2),
                'period_days': days
            }
    
    def get_recent_workouts(self, user_id, limit=10):
        """Get recent workouts for a user."""
        with sqlite3.connect(self.db_file) as conn:
            # Check if created_at column exists
            cursor = conn.execute("PRAGMA table_info(workouts)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'created_at' in columns:
                # Use created_at for more precise ordering
                cursor = conn.execute('''
                    SELECT date, exercise, weight, reps, sets, exertion
                    FROM workouts 
                    WHERE user_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                ''', (user_id, limit))
            else:
                # Fallback to date and id ordering
                cursor = conn.execute('''
                    SELECT date, exercise, weight, reps, sets, exertion
                    FROM workouts 
                    WHERE user_id = ?
                    ORDER BY date DESC, id DESC
                    LIMIT ?
                ''', (user_id, limit))
            
            return cursor.fetchall()

# Initialize database
db = GymDatabase()

# ----------------- Helper Functions -----------------
def categorize_volume(exertion):
    """Categorize daily volume based on exertion score."""
    for category, (min_val, max_val) in VOLUME_THRESHOLDS.items():
        if min_val <= exertion <= max_val:
            return category, VOLUME_CATEGORIES[category]
    return "light", VOLUME_CATEGORIES["light"]

def format_workout_summary(stats):
    """Format workout statistics into a readable string."""
    if stats['exercise_count'] == 0:
        return "No workouts logged today."
    
    category, details = categorize_volume(stats['total_exertion'])
    
    return (
        f"{details['emoji']} Today: {stats['total_exertion']} exertion "
        f"({stats['exercise_count']} exercises)\n"
        f"{details['description']}"
    )

# ----------------- Bot Handlers -----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main menu handler."""
    user = update.effective_user
    daily_stats = db.get_daily_stats(user.id)
    
    welcome_text = f"🏋️‍♂️ **Gym Tracker Bot**\n\n"
    welcome_text += format_workout_summary(daily_stats)
    welcome_text += "\n\nChoose an option:"
    
    keyboard = [
        [InlineKeyboardButton("📝 Log Exercise", callback_data="log_exercise")],
        [InlineKeyboardButton("📊 View Progress", callback_data="view_progress")],
        [InlineKeyboardButton("🎯 Get Recommendations", callback_data="recommend")],
        [InlineKeyboardButton("🔄 Refresh", callback_data="start")]
    ]
    
    try:
        if update.message:
            await update.message.reply_text(
                welcome_text, 
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        elif update.callback_query:
            await update.callback_query.edit_message_text(
                welcome_text, 
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
    except Exception as e:
        logger.error(f"Error in start handler: {e}")
        # Fallback without markdown
        text = welcome_text.replace('**', '')
        if update.message:
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        elif update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    return MAIN_MENU

async def log_exercise_flow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the exercise logging flow."""
    query = update.callback_query
    await query.answer()
    
    # Create exercise group buttons in a 2x3 grid
    keyboard = []
    groups = list(EXERCISES.keys())
    
    # Create rows of 2 buttons each
    for i in range(0, len(groups), 2):
        row = []
        for j in range(2):
            if i + j < len(groups):
                group = groups[i + j]
                row.append(InlineKeyboardButton(
                    group.capitalize(), 
                    callback_data=f"group_{group}"
                ))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("← Back to Main", callback_data="start")])
    
    await query.edit_message_text(
        "Select exercise category:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return EXERCISE_GROUP

async def group_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle exercise group selection."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "start":
        return await start(update, context)
    
    group = query.data.replace("group_", "")
    context.user_data['group'] = group
    exercises = EXERCISES.get(group, [])
    
    # Create exercise buttons in a single column for better readability
    keyboard = []
    for exercise in exercises:
        keyboard.append([InlineKeyboardButton(
            exercise, 
            callback_data=f"exercise_{exercise}"
        )])
    
    keyboard.append([InlineKeyboardButton("← Back", callback_data="log_exercise")])
    
    await query.edit_message_text(
        f"Select {group.capitalize()} exercise:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return EXERCISE_SELECTION

async def exercise_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle exercise selection."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "log_exercise":
        return await log_exercise_flow(update, context)
    
    exercise = query.data.replace("exercise_", "")
    context.user_data['exercise'] = exercise
    
    await query.edit_message_text(
        f"Selected: **{exercise}**\n\n"
        f"Enter weight in kg (enter 0 for bodyweight exercises):",
        parse_mode='Markdown'
    )
    return INPUT_WEIGHT

async def input_weight_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle weight input."""
    try:
        weight = float(update.message.text.strip())
        if weight < 0:
            raise ValueError("Weight cannot be negative")
        
        context.user_data['weight'] = weight
        
        # Create reps selection buttons in a 4x5 grid
        keyboard = []
        for i in range(0, 20, 4):
            row = []
            for j in range(4):
                rep_count = i + j + 1
                if rep_count <= 20:
                    row.append(InlineKeyboardButton(
                        str(rep_count), 
                        callback_data=f"reps_{rep_count}"
                    ))
            keyboard.append(row)
        
        exercise = context.user_data.get('exercise', 'Exercise')
        weight_text = f"{weight}kg" if weight > 0 else "bodyweight"
        
        await update.message.reply_text(
            f"**{exercise}** - {weight_text}\n\nSelect number of reps:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return INPUT_REPS
        
    except ValueError as e:
        await update.message.reply_text(
            "❌ Please enter a valid number for weight (e.g., 80.5 or 0 for bodyweight):"
        )
        return INPUT_WEIGHT

async def input_reps_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle reps input."""
    query = update.callback_query
    await query.answer()
    
    reps = int(query.data.replace("reps_", ""))
    context.user_data['reps'] = reps
    
    # Create sets selection buttons
    keyboard = []
    for i in range(1, 9):
        if i <= 4:
            if i == 1:
                keyboard.append([InlineKeyboardButton(f"{i} set", callback_data=f"sets_{i}")])
            else:
                if len(keyboard) == 0 or len(keyboard[-1]) == 4:
                    keyboard.append([])
                keyboard[-1].append(InlineKeyboardButton(f"{i} sets", callback_data=f"sets_{i}"))
        else:
            if len(keyboard[-1]) == 4:
                keyboard.append([])
            keyboard[-1].append(InlineKeyboardButton(f"{i} sets", callback_data=f"sets_{i}"))
    
    exercise = context.user_data.get('exercise', 'Exercise')
    weight = context.user_data.get('weight', 0)
    weight_text = f"{weight}kg" if weight > 0 else "bodyweight"
    
    await query.edit_message_text(
        f"**{exercise}** - {weight_text} × {reps} reps\n\nSelect number of sets:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    return INPUT_SETS

async def input_sets_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle sets input and log the workout."""
    query = update.callback_query
    await query.answer()
    
    sets = int(query.data.replace("sets_", ""))
    context.user_data['sets'] = sets
    
    # Extract workout data
    user_id = update.effective_user.id
    exercise = context.user_data.get('exercise')
    weight = context.user_data.get('weight', 0)
    reps = context.user_data.get('reps')
    
    # Log the workout
    exertion = db.log_workout(user_id, exercise, weight, reps, sets)
    
    if exertion is not None:
        # Get updated daily stats
        daily_stats = db.get_daily_stats(user_id)
        
        weight_text = f"{weight}kg" if weight > 0 else "bodyweight"
        success_message = (
            f"✅ **Workout Logged!**\n\n"
            f"🏋️ {exercise}: {weight_text} × {reps} × {sets} sets\n"
            f"💪 Exertion: {exertion} points\n\n"
            f"{format_workout_summary(daily_stats)}"
        )
        
        keyboard = [
            [InlineKeyboardButton("➕ Log Another Exercise", callback_data="log_exercise")],
            [InlineKeyboardButton("📊 View Progress", callback_data="view_progress")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="start")]
        ]
        
        try:
            await query.edit_message_text(
                success_message,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except:
            # Fallback without markdown
            await query.edit_message_text(
                success_message.replace('**', '').replace('🏋️', '').replace('💪', ''),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
    else:
        await query.edit_message_text(
            "❌ Failed to log workout. Please try again.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔄 Try Again", callback_data="log_exercise"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="start")
            ]])
        )
    
    return MAIN_MENU

async def view_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show workout progress and statistics."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    # Get daily and weekly stats
    daily_stats = db.get_daily_stats(user_id)
    weekly_stats = db.get_period_stats(user_id, 7)
    recent_workouts = db.get_recent_workouts(user_id, 5)
    
    # Format progress message
    progress_text = "📊 **Your Progress**\n\n"
    
    # Today's stats
    progress_text += f"**Today ({daily_stats['date']}):**\n"
    progress_text += format_workout_summary(daily_stats) + "\n\n"
    
    # Weekly stats
    if weekly_stats['total_exercises'] > 0:
        progress_text += f"**This Week (7 days):**\n"
        progress_text += f"🎯 Total exercises: {weekly_stats['total_exercises']}\n"
        progress_text += f"💪 Total exertion: {weekly_stats['total_exertion']}\n"
        progress_text += f"📈 Average per exercise: {weekly_stats['avg_exertion']}\n\n"
    
    # Recent workouts
    if recent_workouts:
        progress_text += "**Recent Workouts:**\n"
        for workout in recent_workouts[:3]:
            date, exercise, weight, reps, sets, exertion = workout
            weight_text = f"{weight}kg" if weight > 0 else "BW"
            progress_text += f"• {exercise}: {weight_text} × {reps} × {sets} ({exertion} pts)\n"
    else:
        progress_text += "No recent workouts found."
    
    keyboard = [
        [InlineKeyboardButton("📝 Log Exercise", callback_data="log_exercise")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="start")]
    ]
    
    try:
        await query.edit_message_text(
            progress_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except:
        # Fallback without markdown
        clean_text = progress_text.replace('**', '').replace('*', '')
        await query.edit_message_text(
            clean_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    return MAIN_MENU

async def get_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Provide workout recommendations based on user's history."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    daily_stats = db.get_daily_stats(user_id)
    weekly_stats = db.get_period_stats(user_id, 7)
    
    recommendations = "🎯 **Workout Recommendations**\n\n"
    
    total_exertion = daily_stats['total_exertion']
    category, details = categorize_volume(total_exertion)
    
    if category == "light":
        recommendations += (
            "💡 **Suggestions:**\n"
            "• Add 1-2 compound exercises\n"
            "• Focus on major muscle groups\n"
            "• Consider increasing weight or reps\n\n"
            "**Recommended:** Squats, Deadlifts, Bench Press"
        )
    elif category == "moderate":
        recommendations += (
            "💡 **Suggestions:**\n"
            "• Good baseline! Add isolation work\n"
            "• Focus on weak points\n"
            "• Maintain consistency\n\n"
            "**Recommended:** Accessory exercises, Core work"
        )
    elif category in ["heavy", "very_heavy"]:
        recommendations += (
            "💡 **Suggestions:**\n"
            "• Great intensity! Consider recovery\n"
            "• Add mobility work\n"
            "• Plan rest days\n\n"
            "**Recommended:** Light cardio, Stretching"
        )
    else:  # extreme
        recommendations += (
            "💡 **Suggestions:**\n"
            "• Excellent work! Prioritize recovery\n"
            "• Focus on nutrition and sleep\n"
            "• Plan active recovery\n\n"
            "**Recommended:** Rest, Mobility, Light activity"
        )
    
    # Add weekly pattern analysis
    if weekly_stats['total_exercises'] > 0:
        avg_daily = weekly_stats['total_exertion'] / 7
        recommendations += f"\n\n📈 **Weekly average:** {avg_daily:.1f} exertion/day"
        
        if avg_daily < 50:
            recommendations += "\n🔥 **Challenge:** Try to reach 100+ exertion daily!"
        elif avg_daily > 200:
            recommendations += "\n⚖️ **Balance:** Consider adding lighter recovery days"
    
    keyboard = [
        [InlineKeyboardButton("📝 Log Exercise", callback_data="log_exercise")],
        [InlineKeyboardButton("📊 View Progress", callback_data="view_progress")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="start")]
    ]
    
    try:
        await query.edit_message_text(
            recommendations,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except:
        # Fallback without markdown
        clean_text = recommendations.replace('**', '').replace('*', '')
        await query.edit_message_text(
            clean_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel current operation and return to main menu."""
    await update.message.reply_text("❌ Operation cancelled.")
    await asyncio.sleep(1)
    return await start(update, context)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler."""
    logger.error(f"Update {update} caused error {context.error}", exc_info=True)
    
    try:
        if update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ Something went wrong. Please try again or use /start to return to the main menu."
            )
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

# ----------------- Main Application -----------------
def main():
    """Initialize and run the bot."""
    try:
        # Initialize bot application
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Create conversation handler
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("start", start)],
            states={
                MAIN_MENU: [
                    CallbackQueryHandler(start, pattern="^start$"),
                    CallbackQueryHandler(log_exercise_flow, pattern="^log_exercise$"),
                    CallbackQueryHandler(view_progress, pattern="^view_progress$"),
                    CallbackQueryHandler(get_recommendations, pattern="^recommend$"),
                ],
                EXERCISE_GROUP: [
                    CallbackQueryHandler(group_handler)
                ],
                EXERCISE_SELECTION: [
                    CallbackQueryHandler(exercise_handler)
                ],
                INPUT_WEIGHT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, input_weight_handler)
                ],
                INPUT_REPS: [
                    CallbackQueryHandler(input_reps_handler)
                ],
                INPUT_SETS: [
                    CallbackQueryHandler(input_sets_handler)
                ],
            },
            fallbacks=[
                CommandHandler("start", start),
                CommandHandler("cancel", cancel)
            ],
            per_message=False
        )
        
        # Add handlers
        app.add_handler(conv_handler)
        app.add_error_handler(error_handler)
        
        # Start the bot
        logger.info("🏋️‍♂️ Gym Tracker Bot starting...")
        app.run_polling(drop_pending_updates=True)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    main()