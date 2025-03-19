import sqlite3
import pandas as pd


# Connect to the SQLite database
conn = sqlite3.connect("embeddings.db")

# Read the embeddings table into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM embeddings", conn)

# Close the connection
conn.close()


print(
    df
)  # Display the DataFrameconn.close()# Close the connectiondf = pd.read_sql_query("SELECT * FROM embeddings", conn)# Read the embeddings table into a pandas DataFrameconn = sqlite3.connect('embeddings.db')# Connect to the SQLite database
