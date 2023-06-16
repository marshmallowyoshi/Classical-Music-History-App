#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

from dash import Dash, html, dcc, callback, Output, Input, exceptions
import plotly.express as px
import json
import pandas as pd
import re
import numpy as np
import plotly.graph_objects as go
import urllib.request

def mosaic(count_df, epoch_dates, works):
    epoch_start = np.array([min for min, max in epoch_dates])
    labels = [epoch + '<br>(' + str(epoch_dates[i][0]) + ' - ' + str(epoch_dates[i][1]) + ')' for i, epoch in enumerate(epoch_order)]
    labels_bot = ['<br><br>' + labels[i] if i % 2 == 1 else labels[i] for i in range(len(labels))]
    widths = np.array([max - min for min, max in epoch_dates])
    data = count_df.transpose().to_dict('list')

    fig = go.Figure()
    for key in data:
        fig.add_trace(go.Bar(
            name=key,
            y=data[key],
            x = epoch_start - epoch_start[0],
            width = widths,
            offset=0,
            marker_line_width=0,
            customdata = np.transpose([labels,[int(x) for x in (np.round(data[key],1))]]),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}%",
        ))



    fig.update_xaxes(
        tickvals=np.cumsum(widths)-widths/2,
        ticktext = labels_bot,
        fixedrange=True,
        range=[0,epoch_start[-1]-epoch_start[0]+widths[-1]],

        tickfont=dict(size= 15, family='arial', color='white'),
        tickangle=0,
        showgrid=False,
        zeroline=False,

        ticks="outside",
        tickson="boundaries",
        tickcolor='white',

        title="Era (Time in Years)",
        title_font=dict(size=18, family='arial', color='white'),
    )

    fig.update_yaxes(
        range=[0,100],
        fixedrange=True,
        showticklabels= False,
        showgrid=False,
        zeroline=False,

        title_text = "Proportion of Compositions by Genre",
        title_font=dict(size=20, family='arial', color='white'),
        
    )

    fig.update_layout(
        barmode="stack",
        colorway=['#9e3131', '#9e6831', '#9e9e31', '#689e31', '#319e68'],
        xaxis={'automargin': True},
        yaxis={'automargin': True, 'side': 'right'},
        bargap = 0,
        clickmode = "event",
        height = 600,
        margin=dict(l=0, r=10, b=50, t=10),
        legend_title="Genre<br>(Click to filter)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',

        uniformtext=dict(
            mode="hide",
            minsize=10,
        ),

        legend=dict(
            font=dict(
                color="white",
            ),
            itemclick="toggleothers",
            itemdoubleclick=False,
        ),
        )
    return fig

def get_birth(x):
    if x == 'None':
        return None
    else:
        return re.findall(r'\d{4}', x)[0]
def preprocess(df):
    df['composers.birth'] = df['composers.birth'].apply(lambda x: get_birth(x)).astype(int)
    yr_offset = 30

    years = list(range(df['composers.birth'].min()+yr_offset,df['composers.birth'].max()+1+yr_offset))
    out = pd.DataFrame({'year' : years})


    all_genre = df['genre'].value_counts().index

    for genre in all_genre:
        out[genre] = 0
    out['epoch'] = "None"


    for entry in df.iterrows():
        year = entry[1]['composers.birth'] + yr_offset
        genre = entry[1]['genre']

        index = list(out[out['year'] == year].index)
        out.at[index[0],'epoch'] = entry[1]['composers.epoch']
    
        out.at[index[0],genre]+=1

    epoch_fill = "Medieval"
    for entry in out.iterrows():
        if entry[1]['epoch'] != "None":
            epoch_fill = entry[1]['epoch']
        else:
            out.at[entry[0],'epoch'] = epoch_fill

    out = out.set_index('year')

    epoch_grouping = out.groupby('epoch')
    all_counts = { key: list(epoch_grouping[key].sum().reindex(index = epoch_order).values) for key in all_genre }

    count_df = pd.DataFrame(all_counts, index=epoch_order).transpose()
    count_df = 100 *count_df / count_df.sum()

    epoch_dates = [ [out[out['epoch']==epoch].index.min(),out[out['epoch']==epoch].index.max()] for epoch in epoch_order]
    previous = 0
    for index, val in enumerate(epoch_dates):
        epoch_dates[index][0] = previous
        previous = val[1]
    epoch_dates[0][0] = out.index[0]
    return (count_df, epoch_dates)
def works_gen(df):
    works = df.filter(['title','genre','composers.complete_name','composers.epoch'])

    works['metadata'] = works['title'] + '<br>' + works['composers.complete_name']
    works = works.drop(['title','composers.complete_name'], axis=1)

    works = works.groupby(['genre','composers.epoch']).agg({'metadata': lambda x: list(x)})

    works = pd.Series(works.to_dict()['metadata'])
    return works

def load_data(url):
    with urllib.request.urlopen(url) as f:
        data = json.load(f)
    df = pd.json_normalize(data, 
                        record_path=['composers','works'], 
                        meta=[['composers','complete_name'],['composers','epoch'],['composers','birth'],['composers','death']], 
                        meta_prefix=None
                        )
    return df
def load_data_local(filename):
    with open(filename) as f:
        data = json.load(f)
    df = pd.json_normalize(data, 
                        record_path=['composers','works'], 
                        meta=[['composers','complete_name'],['composers','epoch'],['composers','birth'],['composers','death']], 
                        meta_prefix=None
                        )
    return df

# Define Data
df = load_data("https://api.openopus.org/work/dump.json")
# df = load_data_local("dump.json")
epoch_order = ['Medieval','Renaissance','Baroque','Classical','Early Romantic','Romantic','Late Romantic','20th Century','Post-War','21st Century']
works = works_gen(df)

## comment these when doing visualisation from raw data
epoch_dates = [[1165, 1449], [1449, 1661], [1661, 1743], [1743, 1789], [1789, 1832], [1832, 1891], [1891, 1912], [1912, 1951], [1951, 1983], [1983, 2001]]
count_df = pd.read_csv("counts.csv").set_index('Unnamed: 0').reindex(['Vocal', 'Keyboard', 'Chamber', 'Orchestral', 'Stage'])
##

# # Preprocessing (Uncomment to visualise from raw data)
# count_df, epoch_dates = preprocess(df)

# App
app = Dash(__name__)
app.layout = html.Div([
    html.H1(children='The Classical Eras', style={'textAlign':'center'}),
    html.H3(children='Visualising the History of Western Classical Music', style={'textAlign':'center'}),
    dcc.Graph(id='graph-content', figure=mosaic(count_df, epoch_dates, works)),
    html.H2("Dates are a rough measure of the start and end of epochs, there are some overlaps between epochs that are not shown here.", style={'font-size':'11px'}),
    html.H2("Click on a category to get a random composition", style={'bold':'bold'}),
    html.P(id='genre_and_epoch'),
    html.P(id='work', style={'font-size': '18px',}),
    html.A(id='search', children='Youtube Search', href= "https://www.youtube.com/watch?v=PiZRq6sJKeg", target="_blank"),
    html.P(children=" ", style={'textAlign':'center'}),
    html.A(children="Data Source: Open Opus API", style={'textAlign':'center', 'font-size':'10px'}, href= "https://openopus.org"),
])
@callback([
    Output('genre_and_epoch', 'children'),
    Output('work', 'children'),
    Output('search', 'href')],
    [Input('graph-content', 'clickData')]
)
def random_work(clickData):
    if not clickData:
        raise exceptions.PreventUpdate
    
    epochnum = int(clickData["points"][0]['pointNumber'])
    genrenum = int(clickData["points"][0]['curveNumber'])

    epoch = epoch_order[epochnum]
    genre = list(count_df.index)[genrenum]

    worklist = np.random.choice(works.loc[genre,epoch])
    worklist = worklist.split('<br>')
    worklist = " by ".join(worklist)
    
    searchlink = "https://www.youtube.com/results?search_query=" + worklist

    return genre + ' music from the ' + epoch + ' era:', worklist, searchlink
if __name__ == '__main__':
    app.run_server(debug=False)