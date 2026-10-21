import datetime

SKILL = {
    "name": "current_time",
    "description": "Current local date, time, weekday and UTC offset",
    "parameters": {},
}


def run(**_):
    now = datetime.datetime.now().astimezone()
    return {"iso": now.isoformat(timespec="seconds"), "weekday": now.strftime("%A"), "utc_offset": now.strftime("%z")}
