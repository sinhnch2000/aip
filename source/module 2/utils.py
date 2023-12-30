import sqlite3

def check_missing_slots(current_slots, required_slots):
    missing_slots = []
    for slot in required_slots:
        if current_slots[slot] is None:
            missing_slots.append(slot)
    return missing_slots

def generate_query(dict_to_search, current_result = {}):
    query = "SELECT * FROM HOTELS_1 WHERE "
    conditions = []
    for slot, value in dict_to_search.items():
        value = "'{}'".format(value) if type(value) == str else value
        condition = f"{slot} = {value}"
        conditions.append(condition)
    if current_result != {}:
        slot = "hotel_name"
        value = "'{}'".format(current_result[slot])
        condition = f"{slot} != {value}"
        conditions.append(condition)
    query += " AND ".join(conditions)
    print("---query: ", query)
    return query

def select_db(query):
    conn = sqlite3.connect('C:\ALL\OJT\server\gradient_server_test\src\models\module 2\mydatabase.db')
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows