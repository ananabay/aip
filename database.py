import sqlite3
import csv 
import sys

connection = sqlite3.connect('aip.db')
cursor = connection.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS paragraphs (
    id INTEGER PRIMARY KEY,
    paragraph TEXT, 
    book TEXT, 
    author TEXT, 
    dialect TEXT, 
    period TEXT)''')

tsv_name = 'data/preprocessed/all_data_masked.tsv'

csv.field_size_limit(sys.maxsize)

# enter all_data_masked.tsv to sqlite db
def create_db_from_tsv(tsv_name):
    connection1 = sqlite3.connect('aip.db')
    cursor = connection1.cursor()

    with open(tsv_name, 'r') as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        for entry in tsv_reader:
            paragraph, book, author, period, dialect = entry[0], entry[4], entry[3], entry[2], entry[1]
            cursor.execute(
            "INSERT INTO paragraphs (paragraph, book, author, dialect, period) VALUES (?, ?, ?, ?, ?)", (paragraph, book, author, dialect, period))
    connection1.commit()

# given a string, look it up in sqlite db
def check_input_in_db(input_text):
    connection2 = sqlite3.connect('aip.db')
    cursor = connection2.cursor()

    cursor.execute("SELECT * FROM paragraphs WHERE paragraph LIKE ?", ('%' + input_text + '%',))
    result = cursor.fetchall()
    connection2.close()

    return result

# given an entry (text string, author, date, dialect), enter it into sqlite db
def enter_to_db(paragraph, book, author, dialect, period):
    connection3 = sqlite3.connect('aip.db')
    cursor = connection3.cursor()

    cursor.execute(
            "INSERT INTO paragraphs (paragraph, book, author, dialect, period) VALUES (?, ?, ?, ?, ?)", (paragraph, book, author, dialect, period))
    connection3.commit()

if __name__ == "__main__":
    # create_db_from_tsv(tsv_name)
    input_text = ' we should give it a go'
    lookup = check_input_in_db(input_text)
    print(lookup)