import sqlite3


def process_data( data, column_names):
        processed_data = {}
        for i, column_name in enumerate(column_names):
            processed_data[column_name] = [row[i + 1] for row in data]  # Skip the Runid column
        return processed_data

class Computation:
    def __init__():
        con = sqlite3.connect("immunetworks.db")
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS simulator_status (
                            Run_id INTEGER PRIMARY KEY,
                            Status INTEGER CHECK(Status IN (0, 1))
                          )''')
        con.commit()


    def insert_simulation_data( runid, data_dict, type_in):
        con = sqlite3.connect("immunetworks.db")
        cur = con.cursor()
        columns = list(data_dict.keys())
        values = tuple(data_dict.values())
        placeholders = ', '.join(['?' for _ in range(len(values))])
        query = f'INSERT INTO simulator_{type_in}_{runid} (Runid, {", ".join(columns)}) VALUES (?, {placeholders})'
        cur.execute(query, (runid,) + values)
        con.commit()

    def Simulation_status( runid):
        con = sqlite3.connect("immunetworks.db")
        cur = con.cursor()
        cur.execute("SELECT Status FROM simulator_status WHERE Run_id = ?", (runid,))
        status_row = cur.fetchone()
        if status_row is None:
            cur.execute("INSERT INTO simulator_status (Run_id, Status) VALUES (?, 0)", (runid,))
            con.commit()
            return 0  
        else:
            return status_row[0]


    def start_simulation( runid, classes):
        con = sqlite3.connect("immunetworks.db")
        cur = con.cursor()
        columns_str = ', '.join([f'Train_dice_{cls} REAL' for cls in range(classes)])
        query = f'CREATE TABLE IF NOT EXISTS simulator_train_{runid} (Runid Integer,LR REAL, Train_loss REAL, {columns_str})'
        cur.execute(query)
        columns_str = ', '.join([f'Valid_dice_{cls} REAL' for cls in range(classes)])
        query = f'CREATE TABLE IF NOT EXISTS simulator_valid_{runid} (Runid Integer, Valid_loss REAL, {columns_str})'
        cur.execute(query)
        cur.execute("UPDATE simulator_status SET Status = 1 WHERE Run_id = ?", (runid,))
        con.commit()
    
    def end_simulation(runid):
        con = sqlite3.connect("immunetworks.db")
        cur = con.cursor()
        cur.execute("UPDATE simulator_status SET Status = 0 WHERE Run_id = ?", (runid,))
        con.commit()

    def fetch_simulation_data(runid):
        con = sqlite3.connect("immunetworks.db")
        cur = con.cursor()
        fetched_data = {}
        cur.execute(f"PRAGMA table_info(simulator_train_{runid})")
        columns_info = cur.fetchall()
        column_names = [info[1] for info in columns_info if info[1] != 'Runid']
        cur.execute(f"SELECT * FROM simulator_train_{runid} WHERE Runid = ? ORDER BY rowid ASC", (runid,))
        train_data = cur.fetchall()
        fetched_data['train'] = (column_names, train_data)
        cur.execute(f"PRAGMA table_info(simulator_valid_{runid})")
        columns_info = cur.fetchall()
        column_names = [info[1] for info in columns_info if info[1] != 'Runid']
        cur.execute(f"SELECT * FROM simulator_valid_{runid} WHERE Runid = ? ORDER BY rowid ASC", (runid,))
        valid_data = cur.fetchall()
        fetched_data['valid'] = (column_names, valid_data)
        con.close()
        return fetched_data

        
