import datetime

# --------------------------------------------------------------------------------------------------
# Problem 1: Computing the number of days in a month
# --------------------------------------------------------------------------------------------------
def days_in_month(year, month):
    # Determine the start of the next month
    if month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = month + 1
        next_year = year
        
    # Create date objects for the start of the current month and the start of the next month
    current_month_start = datetime.date(year, month, 1) # Changed: date() -> datetime.date()
    next_month_start = datetime.date(next_year, next_month, 1) # Changed: date() -> datetime.date()
    
    # Subtract to find the number of days (timedelta)
    difference = next_month_start - current_month_start
    
    return difference.days

# --------------------------------------------------------------------------------------------------
# Problem 2: Checking if a date is valid
# --------------------------------------------------------------------------------------------------
def is_valid_date(year, month, day):
    
    # 1. Check year range (CRUCIAL: The module name is now correct)
    if not (datetime.MINYEAR <= year <= datetime.MAXYEAR):
        return False
    
    # 2. Check month range
    if not (1 <= month <= 12):
        return False
        
    # 3. Check day range
    if day < 1:
        return False
        
    # 4. Use days_in_month
    max_days = days_in_month(year, month)
    
    if day > max_days:
        return False
        
    return True

# --------------------------------------------------------------------------------------------------
# Problem 3: Computing the number of days between two dates
# --------------------------------------------------------------------------------------------------
def days_between(year1, month1, day1, year2, month2, day2):
    
    # 1. Check validity of both dates using is_valid_date
    if not is_valid_date(year1, month1, day1) or not is_valid_date(year2, month2, day2):
        return 0
        
    # Create date objects
    date1 = datetime.date(year1, month1, day1) # Changed: date() -> datetime.date()
    date2 = datetime.date(year2, month2, day2) # Changed: date() -> datetime.date()
    
    # 2. Check order 
    if date2 < date1:
        return 0
        
    # 3. Calculate difference and return days
    difference = date2 - date1
    
    return difference.days

# --------------------------------------------------------------------------------------------------
# Problem 4: Calculating a person's age in days
# --------------------------------------------------------------------------------------------------
def age_in_days(year, month, day):
    
    # 1. Check validity of the birthday
    if not is_valid_date(year, month, day):
        return 0
        
    # 2. Define the two dates for calculation: Birthday and Today
    birthday = datetime.date(year, month, day) # Changed: date() -> datetime.date()
    today = datetime.date.today() # Changed: date.today() -> datetime.date.today()
    
    # 3. Use days_between.
    age = days_between(birthday.year, birthday.month, birthday.day, 
                       today.year, today.month, today.day)
    
    return age