---
layout: mathjax
title:  Contoso Sale Dashboard
date:   2023-07-20
---


## Sale Data Analysis

In this project, we will build the dashboard and do the data analysis for the Contoso Sale Data.

### Analysis tool

Python libraries of Pandas, Dash, and Plotly for data manipulation and Dashboard development.

### Data Understanding

We have a dataset that shows the sales figures for 2008, 2009, and the characteristics of the products, such as their category, subcategory, store, channel, and location. These are the dimensions of the data.

The diagram below illustrates how the data is related.
![Data relationship](/images/Data-relationship.png)

We found an error in the calculation of the SaleAmount and TotalCost data. They did not account for the quantity of products that were returned by the customers.

The correct formula for the revenue per sale should be as the below calculation:
$$\begin{align} \text{Cost} &= \text{Unit Cost } \times \text{(Sale Quantity - Return Quantity)} \\ \text{Sale} &= \text{Unit Price } \times \text{ (Sale Quantity - Return Quantity)} \\ \text{Profit per sale} & = \text{Sale} - \text{Cost} \end{align}$$
So we should perform the data cleansing accordingly. Other than that, the dataset is ready for the analysis.
```python
# Data Cleansing
FACT2009['Sale_Amount'] = FACT2009['UnitPrice'] * (FACT2009['SalesQuantity'] - FACT2009['ReturnQuantity']) - FACT2009['DiscountAmount']
FACT2009['Cost_Amount'] = FACT2009['UnitCost'] *(FACT2009['SalesQuantity'] - FACT2009['ReturnQuantity'])
FACT2009['Revenue_Per_Sale'] = FACT2009['Sale_Amount'] - FACT2009['Cost_Amount']
```


### Dashboard presentation

I used Pandas to manipulate the data and Dash and Plotly to create the dashboard. This is the result:

![dash board demo](/images/dash_board_demo.gif)

### Data analysis
The comparison of revenue YOY and tracking

![](/images/revenue_copmparison_YOY.png)

![Pie chart](/images/pie_chart_revenue_by_cat.png)
![pie chart](/images/pie_chart_revenue_by_cat.png)
![pie chart](/images/pie_chart_revenue_by_cat_2009.png)
The pie chart shows how much revenue each category contributes. The top three categories are Home Appliances, Computers, and Cameras and camcorders. They generate most of the revenue in both 2008 and 2009. However, we can see some changes in the pie chart. The revenue from Computers increased in 2009, while the revenue from Home Appliances decreased.
More over, it shows that the revenue was lower in 2009 than in 2008 for most of the months. Only in October, the revenue was higher in 2009. This is a clear downward trend in revenue in 2009. We need to explore the reasons for this change.

The group bar chart of revenue grouped by category
![bar chart](/images/bar_chart_revenue_comparison_by_cat.png)
This is clearer in this bar chart as the home appliance revenue decrease.

The bar chart of revenue grouped by sub-category
![bar chart](/images/bar_chart_revenue_comparison_by_subcat.png)
We can see the great drop of Washer and Dryer in the 2009 revenue

### Conclusion
We can easily find out the big drop for the revenue of 2009 is contributed from several product which can share the same property. So we can have the insight what had happened and learn from that.
Beside that, the reduction of revenue in year 2009 may come from the other impact, such as: internal issues, economy issues of financial crisis 2008.

### Appendix of code used

```python
import pandas as pd
import matplotlib.pyplot as plt

DIM = pd.read_excel("1.Lookup_Tables/Contoso_Lookup_Tables.xlsx", sheet_name=["DIM Date", "DIM Product", "DIM Product Category", "DIM Product Sub Category"])

FACT2008 = pd.read_excel("2.Datatables/2008_Contoso_Data.xlsx",sheet_name=["FACT Sales", "FACT Sales Quota"], skiprows=2)

FACT2009 = pd.read_excel("2.Datatables/2009_Contoso_Data.xlsx",sheet_name=["FACT Sales", "FACT Sales Quota"], skiprows=2)

Sale_2008 = FACT2008['FACT Sales']
Sale_2009 = FACT2009['FACT Sales']

# Load the data into Pandas dataframe
## Sale data as FACT eliminate 2 first rows
## Attributes data as DIM

DIM_date = DIM['DIM Date']
DIM_Product = DIM['DIM Product']
DIM_Product_Cat = DIM['DIM Product Category']
DIM_Product_Sub_Cat = DIM['DIM Product Sub Category']

# Data Cleansing 2009
Sale_2009['Sale_Amount'] = Sale_2009['UnitPrice'] * (Sale_2009['SalesQuantity'] - Sale_2009['ReturnQuantity']) - Sale_2009['DiscountAmount']
Sale_2009['Cost_Amount'] = Sale_2009['UnitCost'] *(Sale_2009['SalesQuantity'] - Sale_2009['ReturnQuantity'])
Sale_2009['Revenue'] = Sale_2009['Sale_Amount'] - Sale_2009['Cost_Amount']

# Data Cleansing 2008
## Calculate the revenue

Sale_2008['Sale_Amount'] = Sale_2008['UnitPrice'] * (Sale_2008['SalesQuantity'] - Sale_2008['ReturnQuantity']) - Sale_2008['DiscountAmount']
Sale_2008['Cost_Amount'] = Sale_2008['UnitCost'] *(Sale_2008['SalesQuantity'] - Sale_2008['ReturnQuantity'])
Sale_2008['Revenue'] = Sale_2008['Sale_Amount'] - Sale_2008['Cost_Amount']

# Merge the relationship database
Cat_and_Sub = pd.merge(DIM_Product_Cat, DIM_Product_Sub_Cat, on='ProductCategoryKey')
# Merge the relationship database together
Product_df = pd.merge(DIM_Product, Cat_and_Sub, on="ProductSubcategoryKey")
Sale_2008_df = pd.merge(Sale_2008, Product_df, on="ProductKey")
Sale_2009_df = pd.merge(Sale_2009, Product_df, on="ProductKey")
Sale_2008_df["Month"] = Sale_2008_df["Date"].dt.month
Sale_2009_df["Month"] = Sale_2009_df["Date"].dt.month
data_by_month_2008 = Sale_2008_df.groupby("Month")["Revenue"].sum().reset_index()
data_by_month_2009 = Sale_2009_df.groupby("Month")["Revenue"].sum().reset_index()

# Calculate the percentage change
data_by_month_2009['Percentage_Change'] = ((data_by_month_2009['Revenue'] - data_by_month_2008['Revenue']) / data_by_month_2008['Revenue']) * 100
sale2008_category = Sale_2008_df.groupby("ProductCategoryName")["Revenue"].sum().reset_index()
sale2009_category = Sale_2009_df.groupby("ProductCategoryName")["Revenue"].sum().reset_index()

# Prepare the plot and dashboard
import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Create a Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(children=[
html.H1("Sale Revenue Comparison"),
dcc.Dropdown(
id='year-dropdown',
options=[
{'label': '2008', 'value': '2008'},
{'label': '2009', 'value': '2009'}
],
value='2009', # Default value for the dropdown
),

html.Div([
dcc.Graph(
id='revenue-bar-chart',
figure={}
),

dcc.Graph(
id='pie-chart',
figure={}
)

], style={'display': 'flex', 'flex-direction': 'row'}),

dcc.Dropdown(
id='category-filter',
options=[{'label': category, 'value': category} for category in sale2008_category['ProductCategoryName'].unique()],
value=sale2008_category['ProductCategoryName'].unique()[0],
clearable=False
),

html.Div([
dcc.Graph(
id='cat-revenue-bar-chart',
figure={}
),
dcc.Graph(
id='sub-cat-revenue-bar-chart',
figure={}
)
], style={'display': 'flex', 'flex-direction': 'row'}),


# Callback to update the bar chart
@app.callback(
dash.dependencies.Output('revenue-bar-chart', 'figure'),
[dash.dependencies.Input('revenue-bar-chart', 'relayoutData')]
)

def update_bar_chart(relayoutData):
# Create the grouped bar chart using Plotly
fig = go.Figure()
data_2008 = Sale_2008_df.groupby('Month')['Revenue'].sum().reset_index()
data_2009 = Sale_2009_df.groupby('Month')['Revenue'].sum().reset_index()
fig.add_trace(go.Bar(x=data_2008['Month'], y=data_2008['Revenue'], name='2008'))
fig.add_trace(go.Bar(x=data_2009['Month'], y=data_2009['Revenue'], name='2009'))

# Calculate percentage change in revenue for each month
percentage_change = (data_2009['Revenue'] - data_2008['Revenue']) / data_2008['Revenue'] * 100

# Adding the line trace for percentage change with right-side ticks
fig.add_trace(go.Scatter(x=data_2008['Month'], y=percentage_change, name='% Change in Revenue', mode='lines+markers+text', textposition='top center', yaxis='y2'))
fig.update_layout(
barmode='group',
xaxis_title='Month',
yaxis_title='Sale Revenue $',
yaxis2=dict(title='% Change', overlaying='y', side='right'),
title='Sale Revenue Comparison with Percentage Change',
xaxis=dict(tickmode='linear', dtick=1),
)
return fig
# Callback to update the bar chart
@app.callback(
dash.dependencies.Output('cat-revenue-bar-chart', 'figure'),
[dash.dependencies.Input('cat-revenue-bar-chart', 'relayoutData')]
)

def update_cat_bar_chart(relayoutData):
# Create the grouped bar chart using Plotly
fig = go.Figure()

data_2008_cat_grouped = Sale_2008_df.groupby('ProductCategoryName')['Revenue'].sum().reset_index()
data_2009_cat_grouped = Sale_2009_df.groupby('ProductCategoryName')['Revenue'].sum().reset_index()

fig.add_trace(go.Bar(x=data_2008_cat_grouped['ProductCategoryName'], y=data_2008_cat_grouped['Revenue'], name='2008', marker_color='blue'))

fig.add_trace(go.Bar(x=data_2009_cat_grouped['ProductCategoryName'], y=data_2009_cat_grouped['Revenue'], name='2009', marker_color='orange'))

fig.update_layout(barmode='group', xaxis_title='Month', yaxis_title='Sale Revenue $', title='Sale Revenue Comparison')
fig.update_xaxes(tickmode='linear', dtick=1)

return fig

  

# Define the callback to update the revenue bar chart based on the selected category

@app.callback(
dash.dependencies.Output('sub-cat-revenue-bar-chart', 'figure'),
[dash.dependencies.Input('category-filter', 'value')]
)

def update_revenue_bar_chart(selected_category):
# Filter the data based on the selected category
data_2008_filtered = Sale_2008_df[Sale_2008_df['ProductCategoryName'] == selected_category]
data_2009_filtered = Sale_2009_df[Sale_2009_df['ProductCategoryName'] == selected_category]

# Create the grouped bar chart using Plotly
data_2008_grouped = data_2008_filtered.groupby('ProductSubcategoryName')['Revenue'].sum().reset_index()
data_2009_grouped = data_2009_filtered.groupby('ProductSubcategoryName')['Revenue'].sum().reset_index()

# Create the grouped bar chart using Plotly
fig = go.Figure()

fig.add_trace(go.Bar(x=data_2008_grouped['ProductSubcategoryName'], y=data_2008_grouped['Revenue'], name='2008', marker_color='green'))
fig.add_trace(go.Bar(x=data_2009_grouped['ProductSubcategoryName'], y=data_2009_grouped['Revenue'], name='2009', marker_color='yellow'))

fig.update_layout(
barmode='group',
xaxis_title='Subcategory',
yaxis_title='Total Sale Revenue $',
title=f'Total Sale Revenue Comparison for {selected_category} Subcategories in 2008 and 2009'
)
return fig

# Define the callback to update the pie chart
@app.callback(
dash.dependencies.Output('pie-chart', 'figure'),
[dash.dependencies.Input('year-dropdown', 'value')]
)

def update_pie_chart(selected_year):
# Choose the DataFrame based on the selected year
if selected_year == '2008':
df = Sale_2008_df
elif selected_year == '2009':
df = Sale_2009_df

# Calculate the total revenue for each category
revenue_by_category = df.groupby('ProductCategoryName')['Revenue'].sum().reset_index()

# Create the pie chart
fig = px.pie(revenue_by_category, values='Revenue', names='ProductCategoryName', title=f'Revenue by Category ({selected_year})')
return fig

if __name__ == '__main__':

app.run_server(debug=True)
```


