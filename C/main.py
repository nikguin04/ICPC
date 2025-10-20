import sys

input = sys.stdin.readline

def main():
    start = input().strip()
    end = input().strip()

    if start == end: 
        print("7 days")
        return

    start_day, start_time = start.split()
    end_day, end_time = end.split()
    
    days = ["Mon","Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    start_day = days.index(start_day)
    end_day = days.index(end_day)
    
    start_hour, start_min = map(int, start_time.split(":"))
    end_hour, end_min = map(int, end_time.split(":"))
    
    days = minut = hour = 0
    
    if end_min > start_min:
        minut = end_min - start_min
    elif end_min < start_min:
        minut = (60 - start_min + end_min) % 60
        start_hour = (start_hour+1) % 24
        if start_hour == 0:
            start_day = (start_day + 1) % 7
        
    if end_hour > start_hour:
        hour = end_hour - start_hour
    elif end_hour < start_hour:
        hour = (24 - start_hour + end_hour) % 24
        start_day = (start_day+1) % 7
    
    if end_day > start_day:
        days = (end_day - start_day)
    elif end_day < start_day:
        days = (7 - start_day + end_day) % 7
        
    out = []
    
    if days == 1:
        out += [f"{days} day"]  
    if days > 1:
        out += [f"{days} days"]
    
    if hour == 1:
        out += [f"{hour} hour"]
    if hour > 1:
        out += [f"{hour} hours"]
        
    if minut == 1:
        out += [f"{minut} minute"]
    if minut > 1:
        out += [f"{minut} minutes"]
    
    if len(out) == 2:
         print(" and ".join(out))
    else:
        print(", ".join(out))
    
main()

