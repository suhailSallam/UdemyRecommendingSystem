## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import neattext as ntx
import streamlit as st
from statsmodels.graphics.mosaicplot import mosaic
import streamlit.components.v1 as comp

# Setting page layout ( This command should be after importing libraries )
st.set_page_config(page_title='Recommending System Dashboard',page_icon=None,
                   layout='wide',initial_sidebar_state='auto', menu_items=None)
# Sidebar for input selections
st.sidebar.title('Course Analysis Recommending System')
st.sidebar.write('Use this system for:')
st.sidebar.write('     - Imortant information')
st.sidebar.write('     - Related cources to the course selected')
selectMB = st.sidebar.selectbox('Select Analysis Main Tab', ['Story_telling','Overview','Subjects','Profit', 'Price_Categories_per_Subject', 'Subscribers_per_Subject', 'Courses_per_Subject_per_Year','Recommened_Related_Courses','Recomendations'], key=11)

with st.sidebar:
    st.markdown("""
    <style>
    :root {
      --header-height: 50px;
    }
    .css-z5fcl4 {
      padding-top: 2.5rem;
      padding-bottom: 5rem;
      padding-left: 2rem;
      padding-right: 2rem;
      color: blue;
    }
    .css-1544g2n {
      padding: 0rem 0.5rem 1.0rem;
    }
    [data-testid="stHeader"] {
        background-image: url(/app/static/icons8-astrolabe-64.png);
        background-repeat: no-repeat;
        background-size: contain;
        background-origin: content-box;
        color: blue;
    }

    [data-testid="stHeader"] {
        background-color: rgba(28, 131, 225, 0.1);
        padding-top: var(--header-height);
    }

    [data-testid="stSidebar"] {
        background-color: #e3f2fd; /* I changed the old transparent color to solve overlap between sidebar and main canvas */
        margin-top: var(--header-height);
        color: blue;
        position: fixed; /* Ensure sidebar is fixed */
        width: 250px; /* Fixed width */
        z-index: 999; /* Ensure it stays on top */
    }

    [data-testid="stToolbar"]::before {
        content: "UDEMY - Course Analysis Recommending System";
    }

    [data-testid="collapsedControl"] {
        margin-top: var(--header-height);
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100%; /* Sidebar takes full width on small screens */
            height: auto; /* Adjust height for small screens */
            position: relative; /* Sidebar is not fixed on small screens */
            z-index: 1000; /* Ensure it stays on top */
        }

        .css-z5fcl4 {
            padding-left: 1rem; /* Adjust padding for smaller screens */
            padding-right: 1rem;
        }

        [data-testid="stHeader"] {
            padding-top: 1rem; /* Adjust header padding */
        }

        [data-testid="stToolbar"] {
            font-size: 1.2rem; /* Adjust font size for the toolbar */
        }
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: rgb(30, 103, 119);
   text-align: center; /* Center the text */
   overflow-wrap: break-word;
   font-size: 14px; 
   display: flex;
   flex-direction: column
}

/* breakline for metric text         */
div[data-testid="metric-container"]
> label[data-testid="stMetricLabel"]
> div {
   font-size: 12px; 
   text-align: center;
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: red;
   display: flex;
   flex-direction: column
}
</style>
"""
, unsafe_allow_html=True)

# Cache data loading and processing
@st.cache_data
def load_data():
    df = pd.read_csv('udemy_courses-cleaned1.csv')
    df['published_timestamp'] = pd.to_datetime(df['published_timestamp'])
    df['year'] = df['published_timestamp'].dt.year
    return df

# Load data
df = load_data()

# categorized Data per subject
cat_s_df = df.groupby('subject')['subject'].value_counts()

# categorized Data per subject and price categorty
cat_s_pc_df = df.groupby(['subject','price_category']).size().reset_index(name='Price Category Count')

# Cache course counts calculation
@st.cache_data
def calculate_course_counts():
    return df.groupby(['subject', 'year']).size().reset_index(name='course_count')
course_counts = calculate_course_counts()

# Cache course subjects year calculation
@st.cache_data
def calculate_courses_subjects_year(mySubject_Selected):
    if mySubject_Selected == 'All':
        courses_subjects_data = df
    else:
        courses_subjects_data = df.loc[df['subject']==mySubject_Selected]
    return courses_subjects_data.groupby(['subject', 'year']).size().reset_index(name='course_count')

# Cache Subscribers subjects year calculation
@st.cache_data
def calculate_subscribers_subjects_year(mySubject_Selected):
    if mySubject_Selected == 'All':
        subscribers_subjects_year_data = df
    else:
        subscribers = df.groupby(['subject', 'year']).sum().reset_index()
        subscribers_subjects_year_data = subscribers[subscribers['subject'] == mySubject_Selected]
    return subscribers_subjects_year_data

# prepare some variables
business_finance_data    = df[df['subject']==cat_s_df.index[0][0]]
graphing_design_data     = df[df['subject']==cat_s_df.index[1][0]]
musical_instruments_data = df[df['subject']==cat_s_df.index[2][0]]
web_development_data     = df[df['subject']==cat_s_df.index[3][0]]

# Value of profit per subject
profit_business_finance    = business_finance_data['Profit'].sum()
profit_graphing_design     = graphing_design_data['Profit'].sum()
profit_musical_instruments = musical_instruments_data['Profit'].sum()
profit_web_development     = web_development_data['Profit'].sum()
# count of each price category per subject
price_cat_business_finance = business_finance_data['price_category'].count()
price_cat_graphing_design = graphing_design_data['price_category'].count()
price_cat_musical_instruments = musical_instruments_data['price_category'].count()
price_cat_web_development = web_development_data['price_category'].count()

# Sum of subscribers per subject
subscribers_subject_business_finance = business_finance_data['num_subscribers'].sum()
subscribers_subject_graphing_design = graphing_design_data['num_subscribers'].sum()
subscribers_subject_musical_instruments = musical_instruments_data['num_subscribers'].sum()
subscribers_subject_web_development = web_development_data['num_subscribers'].sum()

# Available subjects
subjects = df['subject'].unique()
labels = [subjects[0],subjects[1],subjects[2],subjects[3]]
Profit_values = [profit_business_finance,profit_graphing_design,profit_musical_instruments,profit_web_development]
# count of each price ]
price_cat_values = [price_cat_business_finance,price_cat_graphing_design,price_cat_musical_instruments,price_cat_web_development]
subscibers_subject_values = [subscribers_subject_business_finance,subscribers_subject_graphing_design,subscribers_subject_musical_instruments,subscribers_subject_web_development]

# define different graphs
# Overview
# Utility function for creating bar charts
def bar_chart(labels, values, myTitle):
    fig, ax = plt.subplots(figsize=(10,3))
    bar_colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']   
    ax.bar(labels, values,color=bar_colors)
    for b,a in zip(values,labels):
        ax.text(a,b,str(b), horizontalalignment="center")
    if len(labels) > 1:
        myTitle = myTitle + 'All' 
    else :
        myTitle = myTitle + labels.values
    plt.title(myTitle)
    plt.xticks(rotation=0, ha='center')
    st.pyplot(fig)

#Profit per Subject
# Scatter
def myScatter1(Scatterdata):
    fig,ax = plt.subplots(figsize=(20,5))
    g = sns.catplot(data=Scatterdata,x=Scatterdata['subject'],y=Scatterdata['profit_subject'],aspect=2.0)
    ax=g.ax
    if len(Scatterdata['subject']) > 1 :
        myTitle = 'SUM Profit of all subjects'
    else :
        myTitle = 'SUM Profit of ' + Scatterdata['subject'].unique()
    ax.set_title(myTitle)
    st.pyplot(g.fig)
# BarPlot
def myBarplot1(data):
    fig, ax = plt.subplots(figsize=(12,5))
    ax = sns.barplot(data=data,x='subject',y='profit_subject',hue='subject')
    ax.set_title('SUM Pofit per Subject')
    i=0
    for i in range(len(data['subject'])):
        ax.bar_label(ax.containers[i])
    st.pyplot(fig)

# Bar plot
def myBarPlot2(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x='year', y='num_subscribers', hue='subject', ax=ax)
    ax.set_title('Number of Subscribers per Subject per Year')
    
    #max_count = data['num_subscribers'].max()
    #new_ylim = (0, max_count * 1.2)
    #ax.set_ylim(new_ylim)
    ax.set_yscale('log')
    ax.legend(loc='best')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
#            ax.text(p.get_x() + p.get_width() / 2, height + (max_count * 0.05),
            ax.text(p.get_x() + p.get_width() / 2, height , f'{height}',
                    ha='center', va='bottom', color='black', fontsize=10)
    st.pyplot(fig)

# Overview, Profit, 
# Pie chart
def PieThis(myList,Subject,values,values_type):
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        pull=[myList[0],myList[1],myList[2],myList[3]])],
                   layout=go.Layout(title=go.layout.Title(
                       text= Subject + ' - ' +values_type),
                                    width=500,
                                    height=500))
    fig.update_layout(template='plotly',title_x=0.36)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Price categories, profit, 
# Bar chart
def barThis(Subject,labels,values):
    fig = px.bar(
        x=labels,
        y=values,
        color=labels,
        title= 'Courses Distribution by ' + Subject,
        width=200, height=393)
    fig.update_layout(template='plotly',title_x=0.45)
    st.plotly_chart(fig, theme=None, use_container_width=True)


# Price Categories
# Scatter
def myScatter():
    fig,ax = plt.subplots(figsize=(20,5))
    g = sns.catplot(data=cat_s_pc_df,x='price_category',y='Price Category Count',hue='subject',aspect=1.2)
    ax=g.ax
    ax.set_title('Price Categoties per Subject')
    st.pyplot(g.fig)
# Bar plot
def myBarplot():
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(data=cat_s_pc_df,x='price_category',y='Price Category Count',hue='subject')
    ax.set_title('Price Categoties per Subject')
    st.pyplot(fig)
# Nested pie chart
def myNested():
    outer = df.groupby('subject').size()
    inner = df.groupby(['subject', 'price_category']).size()
    outer_labels = outer.index
    inner_labels = inner.index.get_level_values(1).unique()
    fig, ax = plt.subplots(figsize=(8,3))
    size = 0.35
    cmap = plt.colormaps["tab20c"]
    outer_colors = cmap(np.arange(4) * 4)
    inner_colors = cmap(np.tile(np.arange(5), 4))
    outer_wedges, _ = ax.pie(outer.values.flatten(), radius=1, colors=outer_colors,
                             wedgeprops=dict(width=size, edgecolor='w'))
    inner_wedges, _ = ax.pie(inner.values.flatten(), radius=1-size, colors=inner_colors,
                             wedgeprops=dict(width=size, edgecolor='w'))
    def place_labels_and_values(wedges, labels, values, radius):
        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            angle_rad = np.deg2rad(angle)
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            horizontal_alignment = 'center'
            vertical_alignment = 'center'
            ax.text(x, y, f"{labels[i]}\n{values[i]}", ha=horizontal_alignment, va=vertical_alignment, fontsize=4, color='black')
    place_labels_and_values(outer_wedges, outer_labels, outer.values, radius=1 - size/2)  # Outer pie
    place_labels_and_values(inner_wedges, inner.index.get_level_values(1), inner.values, radius=1 - size - size/2)  # Inner pie
    unique_outer_labels = outer_labels
    unique_outer_wedges = outer_wedges
    unique_inner_labels = list(inner_labels)
    unique_inner_wedges = []
    seen_inner_labels = set()
    for i, label in enumerate(inner.index.get_level_values(1)):
        if label not in seen_inner_labels:
            unique_inner_wedges.append(inner_wedges[i])
            seen_inner_labels.add(label)
    handles = unique_outer_wedges + unique_inner_wedges
    labels = list(unique_outer_labels) + list(unique_inner_labels)
    ax.legend(handles, labels, title="Legend", loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1),
              prop={'size': 4})  
    ax.set_title('Price Categoties per Subject', fontsize=7)
    ax.set(aspect="equal")
    st.pyplot(fig,theme=None, use_container_width=True)
# Stacked bar chart
def myStacked():
    cat_s_pc_df.reset_index(inplace=True)
    df_pivot = cat_s_pc_df.pivot(index='subject', columns='price_category', values='Price Category Count')
    fig, ax = plt.subplots(figsize=(10, 6))
    df_pivot.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Counts")
    ax.set_title("Stacked Bar Chart by Subject and Price Category")
    st.pyplot(fig)
# Sunburst
def mySunburst():
    fig = px.sunburst(cat_s_pc_df.reset_index(), path=['subject', 'price_category'], values='Price Category Count')
    fig.update_layout(title="Sunburst Chart by Subject and Price Category",
                      width=600, height=500,title_x=0.25)
    fig.update_traces(textinfo="label+percent entry",
                      textfont_size=15 )
    st.plotly_chart(fig, use_container_width=True)
# Treemap
def myTreemap():
    fig = px.treemap(cat_s_pc_df.reset_index(), path=['subject', 'price_category'], values='Price Category Count')
    fig.update_layout(title="Treemap by Subject and Price Category",title_x=0.25)
    st.plotly_chart(fig, use_container_width=True)
# Mosaic chart
def myMosaic():
    fig, ax = plt.subplots(figsize=(10, 6))
    mosaic(cat_s_pc_df.reset_index(), ['subject', 'price_category'],ax=ax)
    plt.title("Mosaic Plot by Subject and Price Category")
    st.pyplot(fig)
# Grouped bar chart
def myGroupedBar():
    df_pivot = cat_s_pc_df.pivot(index='subject', columns='price_category', values='Price Category Count')
    fig, ax = plt.subplots(figsize=(10, 5))
    df_pivot.plot(kind='bar',ax=ax)
    ax.set_ylabel("Counts")
    ax.set_title("Grouped Bar Chart by Subject and Price Category")
    st.pyplot(fig)

# Number of courses per subject per year
# Line plot
def myLinePlot(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=data, x='year', y='course_count', hue='subject')
    plt.title('Number of Courses per Subject per Year')
    for a,b,c in zip(data['year'],data['course_count'],data['subject']):
        plt.text(a,b,str(b), horizontalalignment="center")
    #plt.show()
    st.pyplot(fig)
# Bar plot
def myBarPlot(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='year', y='course_count', hue='subject', ax=ax)
    ax.set_title('Number of Courses per Subject per Year')
    max_count = data['course_count'].max()
    new_ylim = (0, max_count * 1.2)
    ax.set_ylim(new_ylim)
    ax.legend(loc='best')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, height + (max_count * 0.05),
                f'{height}', ha='center', va='bottom', color='black', fontsize=10)
    st.pyplot(fig)

# Scatter plot
def myScatterPlot(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data, x='year', y='course_count', hue='subject', s=100, ax=ax)  # s for size
    ax.set_title('Number of Courses per Subject per Year')
    max_count = data['course_count'].max()
    new_ylim = (0, max_count * 1.2)
    ax.set_ylim(new_ylim)
    for i in range(len(data)):
        row = data.iloc[i]
        ax.text(row['year'], row['course_count'] + (max_count * 0.05),
                f'{row["course_count"]}', ha='center', va='bottom', color='black', fontsize=12, zorder=10)
    st.pyplot(fig)
# Displot
def mydisplot(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    g = sns.displot(data=data, x='year', y='course_count', hue='subject', aspect=1.5)
    ax=g.ax
    for a,b,c in zip(data['year'],data['course_count'],data['subject']):
        ax.text(a,b,str(b), horizontalalignment="center")
    ax.set_title('Number of Courses per Subject per Year')
    st.pyplot(g.fig)

#get related data
def getRelatedData(subject_selected):
    if subject_selected == cat_s_df.index[0][0]:
        sdA = business_finance_data.groupby('price_category')['price_category'].value_counts().sort_index()
        sd = sdA
    elif subject_selected == cat_s_df.index[1][0]:

        sdB = graphing_design_data.groupby('price_category')['price_category'].value_counts().sort_index()
        sd = sdB
    elif subject_selected == cat_s_df.index[2][0]:        
        sdC = musical_instruments_data.groupby('price_category')['price_category'].value_counts().sort_index()
        sd = sdC
    elif subject_selected == cat_s_df.index[3][0]:    
        sdD = web_development_data.groupby('price_category')['price_category'].value_counts().sort_index()
        sd = sdD
    x=[]
    y=[]
    for ndx in range(len(sd)):
        x.append(sd.index[ndx][0])
        y.append(sd.values[ndx])
    return x,y
# get sector_to_pull
def sector_to_pull(subject_selected):
    if   subject_selected == subjects[0]:
         myPull=[0.2,0,0,0]
    elif subject_selected == subjects[1]:
         myPull=[0,0.2,0,0]
    elif subject_selected == subjects[2]:
         myPull=[0,0,0.2,0]
    elif subject_selected == subjects[3]:
         myPull=[0,0,0,0.2]
    return myPull

# Main switch case
class SwitchMCase:
    def case_Story_telling(self):
        st.header('Story Telling')
        st.write('This application is for analysing UDEMY dataset.')
        st.write('in the Overview page, you will find some information about this dataset')
        st.write('the main variant in this data set is the SUBJECT.')
        st.write('I analyzed the profits, courses, price categories and subscirbers per subject and some times per year too')
        st.subheader('Profit / subject')
        st.write('The Web Development courses shows the highest profit and the Musical Instruments shows the lowest profit, so consider expanding the course offering of Web Development and and searching for alternative for Musical Instruments')
        st.subheader('Profit / subject / year')
        st.write('Figure shows a trend of dying for all subjects, there is a problem that need to be addressed, figure shows also that the best performance is for Web Development and the worst performance is for Musical Instruments')
        st.subheader('Profit / year / subject')
        st.write('Figure shows a trend of profit increasing over the years till 2015 which was at the top, then profit started decreasing, the reason should be investigated, Web development had the best performance always')
        st.subheader('Profit / Price categories / subject')
        st.write('Figure shows that (AS for PROFIT), the first best price category is 155:200 with all kind of subjects, the second best price category is 55:100 then 105:150, the worst price category is for price category 20:50')
        st.subheader('Profit / subject / Price categories')
        st.write('Figure shows that (AS for PROFIT), the first best price category is 155:200 with all kind of subjects, the second best price category is 55:100 then 105:150, the worst price category is for price category 20:50')
        st.subheader('Price categories / subject')
        st.write('Figure shows that: - The best price category for all subjects is 20:50 - The 20:50 price category has a maximum in Finance - The 2nd best price category differes between subjects - The Free cources has a percentage of 4% (133 course) in web Development,3% (96 course) in finance,7% (46 course) in Musical Instruments and 6% (35 course) in Graphic design,to be totaled to 8% of all cources (310 course)')
        st.subheader('Subsribers / subject / year')
        st.write('Figuer shows increasing of subscribers over years till 2015, after that the trend is decreasing specially in 2017 an specialy for Musical Instruments')
        st.subheader('Number of courses / subject / year')
        st.write('Figure shows a trend of incraasing number of courses over years starting 2011 till 2016, after that the trend switched to be decreasing, reason should be invistegated')
        return 'StoryTelling Done.'

    def case_Overview(self):
        st.header('Over View')
        st.sidebar.subheader('Over View Setting')
        subject_selected = st.sidebar.selectbox('Select Subject',['All',subjects[0],subjects[1],subjects[2],subjects[3]],key=1)
        figure_type = st.sidebar.selectbox('Select Chart Type', ['Pie Chart', 'Bar Chart','barThis'])
        r11,r12,r13,r14 = st.columns(4)
        r21,r22,r23,r24 = st.columns(4)
        r31, r32, r33, r34 = st.columns(4)
        r11.metric('Columns count in this dataset',len(df.columns))
        r12.metric('Number of Records',len(df))
        r13.metric('Number of totaly null records',df.isnull().sum().values.sum())
        r14.metric('Number of different subjects',len(df.groupby('subject').size()))
        btn1 = r21.button('Show sample data')
        btn2 = r22.button('Show data information')
        if btn1:
            st.write(df.sample(3))
        if btn2:
            st.write(pd.DataFrame({"name": df.columns,
                         "non-nulls": len(df)-df.isnull().sum().values,
                         "nulls": df.isnull().sum().values,
                         "type": df.dtypes.values}))
        
        if figure_type == 'Pie Chart':
            if subject_selected == 'All':
                myPull=[0,0,0,0]
                subject_data = df
                subject_counts = df['subject'].value_counts()
                subject_count = df['subject'].count()
                r31.metric('Subject',subject_selected)
                r32.metric('Number of cources included',subject_count)
                r33.metric('Total Profit made',subject_data['Profit'].sum())
                r34.metric('Number of Subscribers',subject_data['num_subscribers'].sum())
            else:
                subject_data = df[df['subject'] == subject_selected]
                subject_count = subject_data.groupby('subject').size().reset_index(name='count')
                subject_counts = df.groupby('subject').size()
                r31.metric('Subject',subject_selected)
                r32.metric('Number of cources included',subject_count['count'])
                r33.metric('Total Profit made',subject_data['Profit'].sum())
                r34.metric('Number of Subscribers',subject_data['num_subscribers'].sum())
                myPull = sector_to_pull(subject_selected)
            fig = PieThis(myPull,subject_selected,subject_counts,'Course Count')
        elif figure_type == 'Bar Chart':
            if subject_selected == 'All':
                subject_data = df
                subject_counts = df.groupby('subject').size().reset_index(name='count')
                r31.metric('Subject',subject_selected)
                r32.metric('Number of cources included',df['subject'].count())
                r33.metric('Total Profit made',subject_data['Profit'].sum())
                r34.metric('Number of Subscribers',subject_data['num_subscribers'].sum())
            else:
                subject_data = df[df['subject'] == subject_selected]
                subject_counts = subject_data.groupby('subject').size().reset_index(name='count')
                r31.metric('Subject',subject_selected)
                r32.metric('Number of cources included',subject_counts['count'])
                r33.metric('Total Profit made',subject_data['Profit'].sum())
                r34.metric('Number of Subscribers',subject_data['num_subscribers'].sum())
            st.write(subject_counts['subject'])
            st.write(subject_counts['count'])
            bar_chart(subject_counts['subject'], subject_counts['count'], 'Courses Distribution by ')
        elif figure_type == 'barThis':
            if subject_selected == 'All':
                subject_data = df
                subject_counts = df.groupby('subject').size().reset_index(name='count')
                r31.metric('Subject',subject_selected)
                r32.metric('Number of cources included',subject_counts['count'].sum())
                r33.metric('Total Profit made',subject_data['Profit'].sum())
                r34.metric('Number of Subscribers',subject_data['num_subscribers'].sum())
            else:
                subject_data = df[df['subject'] == subject_selected]
                subject_counts = subject_data.groupby('subject').size().reset_index(name='count')
                r31.metric('Subject',subject_selected)
                r32.metric('Number of cources included',subject_counts['count'])
                r33.metric('Total Profit made',subject_data['Profit'].sum())
                r34.metric('Number of Subscribers',subject_data['num_subscribers'].sum())
            fig = barThis(subject_selected,subject_counts['subject'],subject_counts['count'])
        return 'Overview Done.'

    def case_Subjects(self):
        st.header('Main Variant: Subjects')
        st.sidebar.subheader('Subjects Setting')
        r11,r12,r13,r14 = st.columns(4)
        r21,r22,r23,r24 = st.columns(4)
        r31,r32,r33,r34 = st.columns(4)
        r41,r42,r43,r44 = st.columns(4)
        subject_counts = df.groupby('subject').size().reset_index(name='count')
        
        with r11:
            st.subheader(f":blue{[subject_counts['subject'][0]]}")
        with r12:
            st.subheader(f":red{[subject_counts['subject'][1]]}")
        with r13:
            st.subheader(f":green{[subject_counts['subject'][2]]}")
        with r14:
            st.subheader(f":violet{[subject_counts['subject'][3]]}")
            
        r21.metric('Profit',profit_business_finance)
        r22.metric('Profit',profit_graphing_design)
        r23.metric('Profit',profit_musical_instruments)
        r24.metric('Profit',profit_web_development)
        
        r31.metric('Subject count',subject_counts['count'][0])
        r32.metric('Subject count',subject_counts['count'][1])
        r33.metric('Subject count',subject_counts['count'][2])
        r34.metric('Subject count',subject_counts['count'][3])

        r31.metric('Top Subject count',subject_counts['count'].max())
        r32.metric('Top Subject Profit',max(Profit_values))
        r33.metric('Min Subject count',subject_counts['count'].min())
        r34.metric('Min Subject Profit',min(Profit_values))
        return 'Subjects Done.'

    def case_Profit(self):
        selectSB = st.sidebar.selectbox('Select Analysis Sub  Tab', ['per_Subject','per_Subject_per_Year','per_Year_per_Subject','per_Price_Category_per_Subject','per_Subject_per_Price_Category'], key=22)
        S_switcher = SwitchSCase()
        S_result = S_switcher.S_switch(selectSB)
        st.write('Profit: ',S_result)
        return 'Profit Done.'

    def case_Price_Categories_per_Subject(self):
        st.header('Price Categories count per Subject')
        st.sidebar.subheader('Price Categories per Subject Settings')
        subject_selected = st.sidebar.selectbox('Select Subject',[cat_s_df.index[0][0],cat_s_df.index[1][0],cat_s_df.index[2][0],cat_s_df.index[3][0]],key='B')
        figure_type = st.sidebar.selectbox('Select figure type',['Dataset','scatter - catplot','bar plot','nested pie chart','stacked bar chart','sunburst chart','treemap chart','mosaic chart','Grouped bar chart'])
        r1,r2 = st.columns(2)
        if figure_type == 'Dataset':
           with r1: st.write(cat_s_pc_df)
        elif figure_type == 'scatter - catplot':
            with r1: myScatter()
        elif figure_type == 'bar plot':
            with r1: myBarplot()
        elif figure_type == 'nested pie chart':
            with r1: myNested()
        elif figure_type == 'stacked bar chart':
            with r1: myStacked()
        elif figure_type == 'sunburst chart':
            with r1: mySunburst()
        elif figure_type == 'treemap chart':
            with r1: myTreemap()
        elif figure_type == 'mosaic chart':
            with r1: myMosaic()
        elif figure_type == 'Grouped bar chart':
            with r1: myGroupedBar()
        if subject_selected == cat_s_df.index[0][0]:
            x,y = getRelatedData(subject_selected)
            with r2: barThis(subject_selected,x,y)
        elif subject_selected == cat_s_df.index[1][0]:
            x,y = getRelatedData(subject_selected)
            with r2: barThis(subject_selected,x,y)
        elif subject_selected == cat_s_df.index[2][0]:
            x,y = getRelatedData(subject_selected)
            with r2: barThis(subject_selected,x,y)
        elif subject_selected == cat_s_df.index[3][0]:
            x,y = getRelatedData(subject_selected)
            with r2: barThis(subject_selected,x,y)
        comp.html('''
            <div    style="text-align: center;
                    color:blue">
                    Figure shows that:
                        - The best price category for all subjects is 20:50
                        - The 20:50 price category has a maximum in Finance
                        - The 2nd best price category differes between subjects
                        - The Free cources has a percentage of 4% (133 course) in web Development,3% (96 course) in finance,7% (46 course) in Musical Instruments and 6% (35 course) in Graphic design,to be totaled to 8% of all cources (310 course)
            </div>
            ''')

        return 'Price_Categories_per_Subject Done'
    
    def case_Subscribers_per_Subject(self):
        st.header('Subscribers per Subject')
        st.sidebar.subheader('Subscribers per Subject Setting')
        figure_type = st.sidebar.selectbox('Select figure type',['Pie chart','Bar chart'])
        subject_selected = st.sidebar.selectbox('Select Subject',['All',subjects[0],subjects[1],subjects[2],subjects[3]],key=3)
        if subject_selected == 'All':
            myPull = [0,0,0,0]
        else:
            myPull = sector_to_pull(subject_selected)
        if figure_type == 'Pie chart':
            fig = PieThis(myPull,subject_selected,subscibers_subject_values, 'Subsribers')
        elif figure_type == 'Bar chart':
            subscibers_subjects_year_data = calculate_subscribers_subjects_year(subject_selected)
            myBarPlot2(subscibers_subjects_year_data)
        st.write('Figuer shows increasing of subscribers over years till 2015, after that the trend is decreasing specially in 2017 an specialy for Musical Instruments')
        return 'Subscribers_per_Subject Done'

    def case_Courses_per_Subject_per_Year(self):
        st.header('Courses per Subject per Year')
        st.sidebar.subheader('Courses per Subject per Year Setting')
        figure_type = st.sidebar.selectbox('Select figure type',['Line Plot','Bar plot','Scatter Plot','Displot'])
        subject_selected = st.sidebar.selectbox('Select Subject',['All',subjects[0],subjects[1],subjects[2],subjects[3]],key=2)
        if subject_selected == 'All':
            course_subject_data = calculate_courses_subjects_year('All')
        else:
            course_subject_data = calculate_courses_subjects_year(subject_selected)
        if figure_type == 'Line Plot':
            myLinePlot(course_subject_data)
        elif figure_type == 'Bar plot':
            myBarPlot(course_subject_data)
        elif figure_type == 'Scatter Plot':
            myScatterPlot(course_subject_data)
        elif figure_type == 'Displot':
            mydisplot(course_subject_data)
        comp.html('''
            <div    style="text-align: center;
                    color:blue">
                    Figure shows a trend of incraasing number of courses over years starting 2011 till 2016, after that the trend switched to be decreasing, reason should be invistegated
            </div>
            ''')
        return 'case_Courses_per_Subject_per_Year Done.'
    
    def default_case(self):
        return "Default class method executed"

    def M_switch(self, value):
        method_name = f'case_{value}'
        method = getattr(self, method_name, self.default_case)
        return method()

    def case_Recommened_Related_Courses(self):
        st.header('Recommened_Related_Courses')
        st.sidebar.subheader('Recommened_Related_Courses Setting')
        # preparing and cleaning the data
        df = load_data()
        df['course_title'] = df['course_title'].apply(ntx.remove_stopwords)
        df['course_title'] = df['course_title'].apply(ntx.remove_special_characters)
        df = df.dropna().reset_index(drop=True)
        df['course_title_cleaned'] = df['course_title']
        df.drop_duplicates(subset=['course_title_cleaned'],keep = 'first', inplace=True)
        df = df.reset_index(drop=True)
        cv=CountVectorizer()
        title_matrix = cv.fit_transform(df['course_title_cleaned']).toarray()
        sim_matrix = cosine_similarity(title_matrix)
        selectCourse = st.sidebar.selectbox('Select Course Title',df['course_title_cleaned'] )
        r15,r25 = st.columns(2)
        with r15: st.metric('Course Selected',selectCourse)
        searchButton = st.sidebar.button('Search')
        #function to get related cources
        def my_rec_sys(my_title):
            course_index = pd.Series(df.index,index=df['course_title_cleaned'])
            title = my_title
            rec_cources = []
            try:
                scores = list(enumerate(sim_matrix[course_index[title]]))
                sorted_selected_course = sorted(scores, key= lambda x: x[1], reverse=True)
                for i in range(len(sorted_selected_course)):
                     if sorted_selected_course[i][1] <1 and sorted_selected_course[i][1] >= 0.40:
                        rec_cources.append(df['course_title'][sorted_selected_course[i][0]])
            except:
                rec_cources.append('Title Not found')
            rc = pd.DataFrame({'RelatedCources':rec_cources})
            return rc
        if searchButton:
            st.write(my_rec_sys(selectCourse))
        return 'Recommened_Related_Courses Done.'

    def case_Recomendations(self):
        st.header('Recomendations')
        st.write('1. Consider expanding the course offering of Web Development and and searching for alternative for Musical Instruments')
        st.write('2. A trend of dying for all subjects need to be addressed')
        st.write('3. The reason of profit decreasing should be investigated')
        st.write('4. Consider adding more corses to price category 155:200')
        st.write('5. Cosider canceling courses with price category 20:50')
        st.write('6. Free Cources conform 8% of all cources (310 course), cosider adding more to 10% of all cources.')
        st.write('7. Decreasing of  subscribers after 2015 should be investigated.')
        st.write('8. After 2016 the trend of Number ofcurces per subjec pear year seems to be decreasing, reason should be invistegated')
        return 'Recomendations Done.'



# Sub switch case
class SwitchSCase:
    def case_per_Subject(self):
        st.header('What is the total profit made by each subject')
        st.sidebar.subheader('Profit per Subject Setting')
        subject_selected = st.sidebar.selectbox('Select Subject',['All',subjects[0],subjects[1],subjects[2],subjects[3]],key=2)
        figure_type = st.sidebar.selectbox('Select figure type',['scatter - catplot','bar plot','Pie Chart'],key=1)
        if subject_selected == 'All':
            profits_subjects_data = df
            profit_subject_data = profits_subjects_data.groupby('subject').sum()['Profit'].reset_index(name='profit_subject')
            profits_subject_data = df
            if figure_type == 'scatter - catplot':
                myScatter1(profit_subject_data)
            elif figure_type == 'bar plot':
                myBarplot1(profit_subject_data)
            elif figure_type == 'stacked bar chart':
                myStacked()
            elif figure_type == 'sunburst chart':
                mySunburst()
            elif figure_type == 'treemap chart':
                myTreemap()
            elif figure_type == 'mosaic chart':
                myMosaic()
            elif figure_type == 'Grouped bar chart':
                myGroupedBar()
            elif figure_type == 'Pie Chart':
                myPull=[0,0,0,0]
                fig = PieThis(myPull,subject_selected,profit_subject_data['profit_subject'], 'Profit')
        else:
            profits_subjects_data = df[df['subject'] == subject_selected]
            profit_subject_data   = df.groupby('subject')['Profit'].sum().reset_index(name='profit_subject')
            cur_profits_subjects  = df.loc[df['subject']==subject_selected].groupby('subject')['Profit'].sum().reset_index(name='profit_subject')
            if figure_type == 'scatter - catplot':
               myScatter1(cur_profits_subjects)
            elif figure_type == 'bar plot':
               myBarplot1(cur_profits_subjects)
            elif figure_type == 'stacked bar chart':
                myStacked()
            elif figure_type == 'sunburst chart':
                mySunburst()
            elif figure_type == 'treemap chart':
                myTreemap()
            elif figure_type == 'mosaic chart':
                myMosaic()
            elif figure_type == 'Grouped bar chart':
                myGroupedBar()
            elif figure_type == 'Pie Chart':
                myPull = sector_to_pull(subject_selected)
                fig = PieThis(myPull,subject_selected,profit_subject_data['profit_subject'], 'Profit')
        comp.html('''<div style="text-align: center;color:blue">Web Development shows the highest profit. Consider expanding the course offering in this subject</div>
        <div style="text-align: center; color:red">Musical Instruments shows the lowest profit. Conseder searching an other sybject</div>
        ''')

        return 'Profit: per_Subject Done.'

    def case_per_Subject_per_Year(self):
        st.header('What is the total PROFIT made by each SUBJECT per YEAR')
        st.sidebar.subheader('Profit per Subject per year Setting')
        #### How much profit in each subject per year?
        profits = df.groupby(['subject', 'year']).sum().reset_index()
        show = st.sidebar.selectbox('Select What to show',['Dataset','Bar Chart','Pie Chart'],key=1)
        if show == 'Dataset':
            st.write(profits)
        elif show == 'Bar Chart':
            fig = px.bar(data_frame = profits,
                         x = 'subject',
                         y = 'Profit',
                         color = 'year',
                         width =500,
                         height = 530)
            fig.update_xaxes(title='Subject')
            fig.update_yaxes(title='Profit')
            st.plotly_chart(fig, use_container_width=True)
            comp.html('''
                <div    style="text-align: center;
                        color:blue">
                        Figure shows a trend of dying for all subjects, there is a problem that need to be addressed, figure shows also that the best performance is for Web Development and the worst performance is for Musical Instruments
                </div>
                ''')

        elif show == 'Pie Chart':
            st.write('**click any subject to dig detail**')
            fig = px.sunburst(df, path=['subject', 'year'], values='Profit')
            fig.update_layout(title="Sunburst Chart for Profit by Subject and year",
                              width=500, height=500,title_x=0.35)
            fig.update_traces(textinfo="label+percent entry",
                              textfont_size=15 )
            st.plotly_chart(fig, use_container_width=True)
            comp.html('''
                <div    style="text-align: center;
                        color:blue">
                        Figure shows a trend of dying for all subjects, there is a problem that need to be addressed, figure shows also that the best performance is for Web Development and the worst performance is for Musical Instruments
                </div>
            ''')

        return 'Profit: per_Subject_per_Year Done.'

    def case_per_Year_per_Subject(self):
        st.header('What is the total PROFIT made each YEAR per SUBJECT')
        st.sidebar.subheader('Profit per year per Subject Setting')
        #### How much profit in each year per subject?
        profits = df.groupby(['year','subject']).sum().reset_index()
        show = st.sidebar.selectbox('Select What to show',['Dataset','Bar Chart','Pie Chart'],key=2)
        if show == 'Dataset':
            st.write(profits)
        elif show == 'Bar Chart':
            fig = px.bar(data_frame = profits,
                         x = 'year',
                         y = 'Profit',
                         color = 'subject',
                         width =500,
                         height = 530)
            fig.update_xaxes(title='Year')
            fig.update_yaxes(title='Profit')
            st.plotly_chart(fig, use_container_width=True)

        elif show == 'Pie Chart':
            st.write('**click any year to dig detail**')
            fig = px.sunburst(df, path=['year','subject'], values='Profit')
            fig.update_layout(title="Sunburst Chart for Profit by Subject and year",
                              width=500, height=500,title_x=0.35)
            fig.update_traces(textinfo="label+percent entry",
                              textfont_size=15 )
            st.plotly_chart(fig, use_container_width=True)
        comp.html('''
            <div    style="text-align: center;
                    color:blue">
                    Figure shows a trend of profit increasing over the years till 2015 which was at the top, then profit started decreasing, the reason should be investigated, Web development had the best performance always.
            </div>
            ''')

        return 'Profit: per_Year_per_Subject Done.'
    def case_per_Price_Category_per_Subject(self):
        #### How much profit in each price_category per subject?
        profits = df.groupby(['subject', 'price_category']).sum().reset_index()
        fig = px.bar(data_frame = profits,
                     x = 'price_category',
                     y = 'Profit',
                     color = 'subject')
        fig.update_xaxes(title='Price Category')
        fig.update_yaxes(title='Profit')
        st.plotly_chart(fig, use_container_width=True)
            
        comp.html('''
            <div    style="text-align: center;
                    color:blue">
                    Figure shows that (AS for PROFIT), the first best price category is 155:200 with all kind of subjects, the second best price category is 55:100 then 105:150, the worst price category is for price category 20:50
            </div>
        ''')
        
        return 'Profit: Price_Category_per_Subject Done.'


    def case_per_Subject_per_Price_Category(self):
        #### How much profit in each subject per price_category?
        profits = df.groupby(['subject', 'price_category']).sum().reset_index()
        fig = px.bar(data_frame = profits,
                     x = 'subject',
                     y = 'Profit',
                     color = 'price_category')
        fig.update_xaxes(title='Subject')
        fig.update_yaxes(title='Profit')
        st.plotly_chart(fig, use_container_width=True)
        comp.html('''
            <div    style="text-align: center;
                    color:blue">
                    Figure shows that (AS for PROFIT), the first best price category is 155:200 with all kind of subjects, the second best price category is 55:100 then 105:150, the worst price category is for price category 20:50

            </div>
        ''')
        return 'Profit: Subject_per_Price_Category Done.'


    def default_case(self):
        return "Default class method executed"

    def S_switch(self, value):
        method_name = f'case_{value}'
        method = getattr(self, method_name, self.default_case)
        return method()

# Usage
M_switcher = SwitchMCase()
M_result = M_switcher.M_switch(selectMB)
st.write('MAIN: ',M_result)


