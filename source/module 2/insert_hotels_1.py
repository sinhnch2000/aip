import sqlite3

conn = sqlite3.connect('C:\ALL\OJT\server\gradient_server_test\src\models\module 2\mydatabase.db')
cursor = conn.cursor()

insert_query = """
INSERT INTO HOTELS_1 (hotel_name, 
                      destination, 
                      street_address,
                      number_of_rooms_available,
                      star_rating,
                      price_per_night,
                      has_wifi,
                      phone_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""

hotel_1_data = [
    ('CayXanh', "District 1", "21 Nguyen Hue", 23, 4, 12, "True",  "0903610477"),
    ('CayHong', "District 1", "13 Dong Khoi",   1, 5, 20, "False", "0822610477"),
    ('CayVang', "District 3", "23 Nguyen Xien", 2, 3,  8, "True",  "0972334567"),
    ('CayDen',  "District 1", "134 CMT8",       2, 4,  7, "True",  "0972334567"),
]

cursor.executemany(insert_query, hotel_1_data)

conn.commit()
conn.close()