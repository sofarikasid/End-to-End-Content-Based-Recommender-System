from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
from prettytable import PrettyTable


# Create the FastAPI app
app = FastAPI()


# Define an example route that returns a pandas DataFrame
@app.get("/dataframe")
def get_dataframe():
    # Create a sample DataFrame
    data = {
        "Name": ["John", "Alice", "Bob"],
        "Age": [25, 30, 35],
        "City": ["New York", "London", "Paris"]
    }
    df = pd.DataFrame(data)

    # Return the DataFrame as an HTML response
    return df.to_html()


# Define an example route that returns a PrettyTable
@app.get("/prettytable")
def get_prettytable():
    # Create a sample PrettyTable
    table = PrettyTable()
    table.field_names = ["Name", "Age", "City"]
    table.add_row(["John", 25, "New York"])
    table.add_row(["Alice", 30, "London"])
    table.add_row(["Bob", 35, "Paris"])

    # Return the PrettyTable as an HTML response
    return table.get_html_string()


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
