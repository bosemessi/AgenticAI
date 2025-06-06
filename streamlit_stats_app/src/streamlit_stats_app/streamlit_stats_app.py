# Final Agent-created code for the app
#-------------------------------------

import streamlit
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit import file_uploader, selectbox
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

def scatter_plotter():
    # Create a file uploader button to allow users to upload a csv file that contains a mix of numerical and non-numerical columns.
    uploaded_file = streamlit.file_uploader("Select CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Assert that a valid csv file has been uploaded, with atleast two numerical columns. If there are no headers supplied in the csv file,
        # name the numerical columns as Qty1, Qty2, etc. and the non-numerical columns as Str1, Str2, etc.
        if df.columns.size == 0:
            streamlit.write("No CSV data provided")
            return

        try:
            numeric_df = pd.DataFrame(df.select_dtypes(include=[np.number]))
            numeric_df.columns = ['Qty' + str(x) for x in range(numeric_df.shape[1])]
        except ValueError as e:
            streamlit.write('Error: {}'.format(e))
            return

        try:
            non_numeric_df = pd.DataFrame(df.select_dtypes(exclude=[np.number]))
            non_numeric_df.columns = [str(x) for x in range(non_numeric_df.shape[1])]
        except ValueError as e:
            streamlit.write('Error: {}'.format(e))
            return

        # Create two dropdown selection buttons named as "Select Qty 1" and "Select Qty 2".
        selected_qty_1 = selectbox("Select Qty 1", numeric_df.columns, index=0)
        selected_qty_2 = selectbox("Select Qty 2", numeric_df.columns, index=1)

        # Once the csv file has been read into a dataframe, create a scatter plot of the two selected quantities.
        fig = px.scatter(numeric_df, x=selected_qty_1,
                         y=selected_qty_2,
                          template="plotly_white", size_max=15,
                          title=f"Scatter Plot of Qty {selected_qty_1} vs Qty {selected_qty_2}")

        # Add a color bar to the plot.
        # fig.update_layout(coloraxis=dict(ticks='outside'))

        # Add hover text for each point
        # fig.update_traces(hovertext=[str(numeric_df.loc[i, selected_qty_1]) + '\n' + str(non_numeric_df.loc[i, 'Str 1']) for i in range(len(numeric_df))])

        streamlit.plotly_chart(fig)

scatter_plotter()