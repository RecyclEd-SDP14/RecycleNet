import mysql.connector
from mysql.connector.constants import ClientFlag
from datetime import datetime

config = {
    'user': 'root',
    'password': 'parkmerlin123',
    'host': '35.233.62.118',
    'client_flags': [ClientFlag.SSL],
    'database': 'recycled-database'
}


def connect_DB(buffered_val=False):
    cnxn = mysql.connector.connect(**config)
    if buffered_val:
        return cnxn, cnxn.cursor(buffered=True)
    else:
        return cnxn, cnxn.cursor()


def disconnect_DB(cnxn, cursor):
    cursor.close()
    cnxn.close()


def insert_test_fields():
    cnxn, cursor = connect_DB(True)
    cursor.execute("DESCRIBE recycling")
    now = datetime.now()
    formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
    sql = "INSERT INTO recycling (device_name,recycling_date,refund_value,no_plastic_bottles,no_glass_bottles,no_cans,no_trash) VALUES(%s,%s,%s,%s,%s,%s,%s)"
    val = [
        ('test_device_1', formatted_date, 40, 1, 1, 0, 2),
        ('test_device_1', formatted_date, 80, 1, 1, 2, 3),
        ('test_device_2', formatted_date, 60, 1, 1, 1, 0),
        ('test_device_2', formatted_date, 20, 1, 0, 0, 0),
    ]
    cursor.executemany(sql, val)
    cnxn.commit()
    disconnect_DB(cnxn, cursor)


def print_all():
    cnxn, cursor = connect_DB()
    cursor.execute("SELECT * FROM recycling")
    for x in cursor:
        print(x)
    disconnect_DB(cnxn, cursor)


def print_recycling_totals_per_device(device):
    cnxn, cursor = connect_DB()
    cursor.execute(
        "SELECT device_name, sum(no_glass_bottles),sum(no_plastic_bottles),sum(no_cans),sum(no_trash) FROM recycling WHERE device_name = \'" + device + "\'")
    for x in cursor:
        result = x
    disconnect_DB(cnxn, cursor)
    return result


def print_refund_value_per_device(device):
    cnxn, cursor = connect_DB()
    cursor.execute("SELECT device_name, sum(refund_value) FROM recycling WHERE device_name = \'" + device + "\'")
    for x in cursor:
        result = x
    disconnect_DB(cnxn, cursor)
    return result


def print_by_device():
    cnxn, cursor = connect_DB()
    cursor.execute("SELECT * FROM recycling GROUP BY device_name ")
    for x in cursor:
        print(x)
    disconnect_DB(cnxn, cursor)
