import sqlite3
conn = sqlite3.connect('C:\ALL\OJT\server\gradient_server_test\src\models\module 2\mydatabase.db')
cursor = conn.cursor()

drop_table_query = "DROP TABLE HOTELS_1"
cursor.execute(drop_table_query)

create_table_query = '''
CREATE TABLE HOTELS_1 (
    hotel_name TEXT PRIMARY KEY,
    destination TEXT,
    street_address TEXT,
    number_of_rooms_available INTEGER,
    star_rating INTEGER,
    price_per_night INTEGER,
    has_wifi TEXT,
    phone_number TEXT
);
'''
cursor.execute(create_table_query)
conn.commit()
conn.close()