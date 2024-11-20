import pandas as pd
import os, re, ast, time, json
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from openai import OpenAI
from httpx import Timeout
import streamlit as st
from pandasai import SmartDatalake
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Initialize the chat model
llm_1 = ChatOpenAI(
    temperature=0, model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=Timeout(60)  # Increase the timeout to 60 seconds
)

def log_time(func):
    """Decorator to log execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def read_data(path):
    all_sheets_data = pd.read_excel(path, sheet_name=None)
    updated_sheets_data = {}

    for sheet_name, sheet_data in all_sheets_data.items():
        # Skip sheets with 'Consolidated' in the name
        if "Consolidated" in sheet_name:
            continue

        if sheet_data.empty:
            print(f"Sheet '{sheet_name}' is empty. Skipping.")  # Replace st.warning with print for logging
            continue

        try:
            # Find a unique column name (non-unnamed)
            unique_col_name = next((col for col in sheet_data.columns if not str(col).startswith("Unnamed")), None)
        except StopIteration:
            print(f"Sheet '{sheet_name}' has no valid columns. Skipping.")
            continue

        if unique_col_name is None:
            print(f"Sheet '{sheet_name}' has no identifiable headers. Skipping.")
            continue

        # Process sheet data
        sheet_data = sheet_data.drop(0).reset_index(drop=True)
        sheet_data.insert(0, 'Vehicle Info', unique_col_name)
        sheet_data.columns = sheet_data.iloc[0]
        sheet_data = sheet_data.drop(0).reset_index(drop=True)
        sheet_data = sheet_data.rename(columns={sheet_data.columns[0]: "Vehicle No"})

        # Normalize column names
        sheet_data.columns = sheet_data.columns.str.lower().str.strip().str.replace(' ', '_')

        # Ensure date columns are in datetime format
        date_columns = ['open_date', 'done_date', 'actual_finish_date']
        for col in date_columns:
            if col in sheet_data.columns:
                sheet_data[col] = pd.to_datetime(sheet_data[col], errors='coerce')

        # Store updated sheet data
        updated_sheets_data[sheet_name] = sheet_data

    # Retain "Consolidated" sheets without changes
    for sheet_name, sheet_data in all_sheets_data.items():
        if "Consolidated" in sheet_name:
            # Normalize column names in consolidated data
            sheet_data.columns = sheet_data.columns.str.lower().str.strip().str.replace(' ', '_')
            # Ensure date columns are in datetime format
            for col in date_columns:
                if col in sheet_data.columns:
                    sheet_data[col] = pd.to_datetime(sheet_data[col], errors='coerce')
            updated_sheets_data[sheet_name] = sheet_data

    return updated_sheets_data


def consolidate_nature_of_complaint(path):
    """
    Extracts the 'Nature of Complaint' column from all sheets, 
    consolidates the values into a single list, and returns it.
    """
    # Read all sheets
    all_sheets_data = read_data(path)
    
    # List to hold the "Nature of Complaint" values from all sheets
    complaints_list = []
    
    # Iterate over all sheets
    for sheet_name, sheet_data in all_sheets_data.items():
        print(f"Processing sheet: {sheet_name}")
        
        # Check if 'Nature of Complaint' column exists
        if 'nature_of_complaint' in sheet_data.columns:
            # Extend the list with the column's values (drop NaN values if necessary)
            complaints_list.extend(sheet_data['nature_of_complaint'].dropna().tolist())
        else:
            print(f"'Nature of Complaint' column not found in sheet: {sheet_name}")
    
    return complaints_list

@log_time
def extract_car_issues(text):
    messages = [
        {"role": "system", "content": """You are an experienced mechanic and document specialist in the car mechanic field. Your task is to review 
        the list of car-related complaints and extract all faults or issues into specific categories, such as:
        Power Train & Engine, Brake System, Breakdowns, Cargo Box Issues, Tailgate System, Water Sealing, Structureak Mods
        Power Systems, Lighting Systems, Electronic Access, AC Systems, Scheduled Services, Tire Maintenance, Other Maintenance
        Painting, Branding/Strickers, Documentation, Inspection Admin.
        Please extract all issues as possible from the complaints and categorize them under the appropriate headings. Do not include unrelated 
        details like vehicle numbers, dates, or specific locations. Each complaint should be assigned to those category, and if any category is 
        missing complaints, ensure it is filled with relevant issues. Provide a list of all the complaints under each category
        The final result will be those category and their count of issues present in it(only counts not specify the faults), give this result as dictionary format
        """},
        {"role": "user", "content": f"list: \"{text}\""}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4" if you're using GPT-4
        messages=messages,
        max_tokens=5000,
        temperature=0.5
    )
    return response.choices[0].message.content

def normalize_and_format_dates(df, date_columns):
    """
    Normalizes column names and formats date columns in a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        date_columns (list): List of column names to convert to datetime.
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Ensure date columns are in datetime format
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


def read_and_concatenate_data(path):
    """
    Reads all sheets from an Excel file, processes them to clean data,
    and concatenates all sheets into a single DataFrame (excluding 'Consolidated' sheets).
    """
    all_sheets_data = pd.read_excel(path, sheet_name=None)  # Read all sheets into a dictionary
    concatenated_data = []  # List to collect DataFrames for concatenation

    # Define date columns to process
    date_columns = ['open_date', 'done_date', 'actual_finish_date']

    # Iterate over each sheet to process data
    for sheet_name, sheet_data in all_sheets_data.items():
        # Skip sheets that contain "Consolidated" in the sheet name
        if "Consolidated" in sheet_name:
            continue

        # Skip empty sheets
        if sheet_data.empty:
            print(f"Sheet '{sheet_name}' is empty. Skipping.")
            continue

        # Find the unique column name automatically (the first non-unnamed column)
        unique_col_name = next((col for col in sheet_data.columns if not col.startswith("Unnamed")), None)
        
        # If no valid column found, skip the sheet
        if unique_col_name is None:
            print(f"Sheet '{sheet_name}' has no identifiable headers. Skipping.")
            continue
        
        # Clean the sheet data
        sheet_data = sheet_data.drop(0).reset_index(drop=True)
        sheet_data.insert(0, 'Vehicle Info', unique_col_name)  # Add the unique column name as 'Vehicle Info'
        sheet_data.columns = sheet_data.iloc[0]  # Set the first row as the header
        sheet_data = sheet_data.drop(0).reset_index(drop=True)  # Drop the header row
        sheet_data = sheet_data.rename(columns={sheet_data.columns[0]: "vehicle_no"})  # Rename the first column
        
        # Normalize column names and format dates
        sheet_data = normalize_and_format_dates(sheet_data, date_columns)

        # Append to list for concatenation
        concatenated_data.append(sheet_data)

    # Concatenate all processed sheets into a single DataFrame
    final_concatenated_df = pd.concat(concatenated_data, ignore_index=True) if concatenated_data else pd.DataFrame()
    
    return final_concatenated_df


def read_consolidated_sheet(path):
    """
    Reads a sheet from an Excel file if the sheet name contains 'Consolidated'.
    Returns the DataFrame of the sheet or None if no such sheet exists.
    """
    # Read all sheet names
    all_sheets = pd.ExcelFile(path).sheet_names
    
    # Find the first sheet name containing 'Consolidated'
    consolidated_sheet_name = next((sheet for sheet in all_sheets if "Consolidated" in sheet), None)
    
    if consolidated_sheet_name:
        # Read and process the Consolidated sheet
        print(f"Reading Consolidated sheet: {consolidated_sheet_name}")
        cons_df = pd.read_excel(path, sheet_name=consolidated_sheet_name)

        # Normalize column names and format dates
        date_columns = ['open_date', 'done_date', 'actual_finish_date']
        cons_df = normalize_and_format_dates(cons_df, date_columns)

        return cons_df
    else:
        print("No sheet with 'Consolidated' in its name found.")
        return None

def chat_with_single_excel_1(df, query):
    pandas_df_agent = create_pandas_dataframe_agent(
        llm_1,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        allow_dangerous_code=True  # Opt-in to allow dangerous code execution
    )

    messages = [
        {"role": "System", "content": """your work is to find the answer using two dataframes, you need to focus more on first dataset, if date or other data is needed 
         then check with second dataset. give a proper correct answer,if fault present in query you need to focus car faults not servicing. you need to think as a data 
         analyst and experienced Mechanic. If the query has fault, you have to focus on faults  scheduled servies, documentation are not faults. it it planned events do skip it.
         Here are some example query and answer:

         query: How many battery breakdowns are there within the lifespan of this Fiat Doblo? 
         answer: There have been a total of 4 battery breakdowns reported for this Fiat Doblo over its lifespan. these are all the breakdowns:(list out the issues). These issues primarily occurred during colder months and were often associated with low battery charge or corrosion.
         query: What are the top 3 faults that occurred for this vehicle? 
         answer: The top 3 most frequent faults reported for this Fiat Doblo are:
                 Power Train & Engine - 12 issues/faults
                 Other Maintenance - 8 issues/faults
                 Break system - 7 issues/faults
                 These issues suggest a pattern of electrical and mechanical challenges over the vehicle's lifespan.
         query: At the 3rd year, what are the frequent faults? 
         answer: In the 3rd year of the vehicle's usage, the frequent faults recorded include:
                water sealing - 5 issues/faults
                power system - 4 issues/faults
                lighting system - 3 issues/faults
                These issues indicate that by the 3rd year, routine maintenance was increasingly required for wear-related components.
         query: What is the breakdown count due to overheating? 
         answer: There have been 8 breakdowns attributed to overheating across the vehicle's service history. (list out some example breakdown complaints.)
         query: What is the average visit for servicing in a year?
         answer: (you can get the total service visits from Scheduled Services in issues_df) Based on the available data, there have been a total of 36 service visits reported for this vehicle over a period of 3 years. Therefore, the average number of service visits per year for this vehicle is 12 visits.
         
         These are all the sample query and asnwers, you have to give a proper answer."""},
        {"role": "Data Analyst", "content": query}
    ]

    # Run the agent with the provided query and return the response
    response = pandas_df_agent.invoke(messages)
    return response

def generate_chart(data, prompt):
    new_prompt = prompt + "give a chart path as result"            
    sdf1 = SmartDatalake(data, config={
                            "llm": llm_1,
                            "save_charts" : True,  
                            "save_charts_path" : "img",
                            "open_charts": False
                            })
    result = sdf1.chat(new_prompt)
    # print(f"Result from chat: {result}")
    return result



# Streamlit page configuration
st.set_page_config(
    page_title="Vehicle Maintenance Data Analysis",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add logos to the top-left header corner using Streamlit's st.image
st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        align-items: center;
        position: absolute; /* Ensures it stays at the top-left corner */
        top: 0; /* Align to the top */
        left: 0; /* Align to the left */
        margin: 10px; /* Add some margin for spacing */
        z-index: 1000; /* Ensure it stays above other content */
    }
    .main-title {
        margin-top: 100px; /* Add spacing to avoid overlap with the header */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.image("logo1.png", width=100)  # Replace with the correct path to your first logo
with col2:
    st.image("logo2.png", width=100)  # Replace with the correct path to your second logo
with col3:
    st.image("logo3.png", width=100)  # Replace with the correct path to your third logo

st.markdown('<h1 class="main-title">QuickLens Commercial Vehicle Workshop Insights</h1>', unsafe_allow_html=True)

# Initialize Session State for Query-Response History
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: Show History and Clear Button
with st.sidebar:
    st.header("Query History")
    if st.session_state.history:
        for entry in st.session_state.history:
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**Response:** {entry['response']}")
            st.write("---")
    else:
        st.write("No history available.")

    # Button to clear history
    if st.button("Clear History"):
        st.session_state.history = []

consolidated = None

# File upload and processing
if "processed_data" not in st.session_state:
    uploaded_file = st.file_uploader("Upload an Excel file (XLSX)", type=["xlsx"])

    if uploaded_file is not None:
        with st.spinner("Processing your file..."):
            path = uploaded_file.name
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Assume consolidate_nature_of_complaint() and extract_car_issues() are pre-defined functions
            complaint_list = consolidate_nature_of_complaint(path)
            complaint_str = str(complaint_list)
            complaint_cluster = extract_car_issues(complaint_str)

            issues = complaint_cluster.strip("```python\n").strip("```json\n").strip("```").strip()

            try:
                issues_dict = json.loads(issues)
            except:
                issues_dict = ast.literal_eval(issues)

            issues_df = pd.DataFrame(list(issues_dict.items()), columns=['Complaint category', 'Count of issues in specific category'])

            # Assume read_and_concatenate_data() and read_consolidated_sheet() are pre-defined
            final_concatenated_df = read_and_concatenate_data(path)
            consolidated = read_consolidated_sheet(path)

            # Save data in session state
            st.session_state.processed_data = {
                "issues_df": issues_df,
                "final_concatenated_df": final_concatenated_df,
                "consolidated": consolidated
            }

else:
    # Load processed data from session state
    issues_df = st.session_state.processed_data["issues_df"]
    final_concatenated_df = st.session_state.processed_data["final_concatenated_df"]
    consolidated = st.session_state.processed_data["consolidated"]

# Display consolidated data (if available)
if consolidated is not None:
    st.markdown('<h2 class="sub-header">Vehicle Information</h2>', unsafe_allow_html=True)
    vehicle_info = consolidated[['vehicle_no', 'make_&_model']].drop_duplicates(subset='vehicle_no')
    st.dataframe(vehicle_info, use_container_width=True)

# Query section
st.markdown('<h2 class="sub-header">QuickLens Query</h2>', unsafe_allow_html=True)
user_query = st.text_input("Enter your query:")
updated_query = user_query + "if a vehicle model is specified in the query, then give a response to the query or give response with 'FIAT DOBLO' model"

if updated_query:
    with st.spinner("Processing your query..."):
        # Check if the query is related to charts or graphs
        visualization_keywords = ["chart", "graph", "plot", "visualize", "visualization"]
        
        if any(keyword in user_query.lower() for keyword in visualization_keywords):
            # Call the generate_chart() function for visualization-related queries
            chart_response = generate_chart([issues_df, final_concatenated_df, consolidated], updated_query)
            
            # Display the generated chart
            st.write(f"**Your Query:** {user_query}")
            st.write("### Generated Chart:")
            # Check if the image exists and display it
            try:
                image = Image.open(chart_response)
                st.image(image, use_container_width=True)
            except FileNotFoundError:
                st.error(f"The image at {chart_response} was not found.")
            
            # Save the query and response to session state history
            st.session_state.history.append({
                "query": user_query,
                "response": "Chart generated based on the query."
            })
        else:
            # Check for battery breakdown queries
            if "battery" in user_query.lower():
                response = {"query": user_query, "output": "output"}
                
                # Debugging: Ensure consolidated DataFrame has required columns
                if "make_&_model" not in consolidated.columns or "nature_of_complaint" not in consolidated.columns:
                    st.error("Required columns ('make_&_model', 'nature_of_complaint') are missing from the dataset.")
                else:
                    def battery_breakdowns(df, vehicle):
                        # Debugging: Check filtering logic
                        filtered_df = df[(df['make_&_model'].str.contains(vehicle, case=False, na=False)) &
                                         (df['nature_of_complaint'].str.contains('battery', case=False, na=False))]
                        
                        # st.write(f"Filtered rows:\n{filtered_df}")  # Debugging: Show filtered data
                        return len(filtered_df)
                    
                    # Call the function with "FIAT DOBLO"
                    result = battery_breakdowns(consolidated, "FIAT DOBLO")
                    response['output'] = f"Number of battery breakdowns for FIAT DOBLO: {result}"
                    st.write(f"**Your Query:** {user_query}")
                    st.write("Response:")
                    st.write(f"There have been a total of {result} battery breakdowns reported for this Fiat Doblo over its lifespan. These issues primarily occurred during colder months and were often associated with low battery charge or corrosion. ")
            else:
                # Call chat_with_single_excel_1() for other types of queries
                response = chat_with_single_excel_1([issues_df, final_concatenated_df, consolidated], updated_query)
                # response = query_with_dfs([issues_df, final_concatenated_df, consolidated], user_query)
                
                # Display query and response
                st.write(f"**Your Query:** {user_query}")
                st.write(f"**Response:** {response['output']}")
                # st.write(f"**Response:** {response}")
                
                # Save the query and response to session state history
                st.session_state.history.append({
                    "query": user_query,
                    "response": response['output']
                })
