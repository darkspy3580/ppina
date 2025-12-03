# IMPORTING LIBRARIES
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import time
import matplotlib.pyplot as plt
from networkx.algorithms.community.centrality import girvan_newman
import base64  # Added for filedownload function
import os
from Bio import Entrez

# SETTING THE LAYOUT AND GIVE PAGE TITLE AND ICON FOR THE APP
st.set_page_config(
    page_title='PPINA : Protein-Protein Interaction Network Analysis', 
    page_icon='a.jpg', 
    layout="wide"
)

def get_tax_id(species):
    """to get data from ncbi taxonomy, we need to have the taxid. we can
    get that by passing the species name to esearch, which will return
    the tax id"""
    species = species.replace(' ', "+").strip()
    search = Entrez.esearch(term=species, db="taxonomy", retmode="xml")
    record = Entrez.read(search)
    return record['IdList'][0]

def get_tax_data(taxid):
    """once we have the taxid, we can fetch the record"""
    search = Entrez.efetch(id=taxid, db="taxonomy", retmode="xml")
    return Entrez.read(search)

@st.cache_data
def convert_df(data1):
    return data1.to_csv().encode('utf-8')

# DOWNLOAD AS CSV FILE
def filedownload(df):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ppi_data.csv">Download csv file</a>'
    return href

# CREATING THE SIDEBAR
with st.sidebar:
    selected1 = option_menu("Home", 
        ["Introduction", 'Tutorial', 'PPI Network Construction', 'NCBI Organism ID Finder', 'Sample Dataset', 'Contact Us'], 
        icons=['lightbulb', 'book', 'bezier', '123', 'book', 'telephone'], 
        menu_icon="house-door", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "17px"}, 
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "black"},
        }
    )

# Initialize selected2 with a default value
selected2 = None

# PPI NETWORK CONSTRUCTION
if selected1 == "PPI Network Construction":
    with st.sidebar:
        selected2 = option_menu(
            "PPI Network Construction", 
            ["Query a Single Protein", 'Query Multiple Proteins', 'Upload Edge List of Protein Interactions'], 
            icons=['patch-question', 'patch-question', 'file-arrow-up'], 
            menu_icon="bezier", 
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "17px"}, 
                "nav-link": {"font-size": "17px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "black"},
            }
        )

if selected1 == "Sample Dataset":
    # Apply custom CSS styles for the entire section
    st.markdown("""
        <style>
            /* Card-like styling for dataframes */
            .dataset-card {
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 30px;
                background-color: white;
            }
            
            /* Custom styling for headers */
            .dataset-header {
                color: #2C3E50;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: 600;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #3498DB;
            }
            
            /* Styling for dataframes */
            .stDataFrame {
                border: none !important;
            }
            
            /* Download button styling */
            .stDownloadButton button {
                background-color: #3498DB !important;
                color: white !important;
                border-radius: 5px !important;
                border: none !important;
                padding: 8px 16px !important;
                font-weight: 500 !important;
                transition: all 0.3s !important;
            }
            
            .stDownloadButton button:hover {
                background-color: #2980B9 !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
            }
            
            /* Info text styling */
            .info-text {
                font-size: 14px;
                color: #7F8C8D;
                margin-bottom: 15px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Page header
    st.markdown("<h1 style='text-align: center; color: #3498DB; margin-bottom: 30px;'>Coronavirus Sequence Datasets</h1>", unsafe_allow_html=True)
    
    # Brief description
    st.markdown("<p class='info-text'>The following datasets contain spike protein sequences from different coronavirus strains affecting humans. These datasets are crucial for comparative genomic analysis and studying viral evolution.</p>", unsafe_allow_html=True)
    
    # SARS-CoV Dataset
    st.markdown("<div class='dataset-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='dataset-header'>SARS-CoV Spike Protein Sequences</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Severe Acute Respiratory Syndrome Coronavirus, first identified in 2003.</p>", unsafe_allow_html=True)
    
    df1 = pd.read_csv("S-human SARS-CoV.csv")
    st.dataframe(df1.style.set_properties(**{
        'background-color': '#f8f9fa', 
        'color': '#2C3E50',
        'border': '1px solid #eaeaea',
        'font-family': 'Segoe UI, Arial, sans-serif'
    }).highlight_max(axis=0, color='#E1F5FE'))
    
    csv1 = df1.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download SARS-CoV Dataset", 
        data=csv1, 
        file_name="S-human_SARS-CoV.csv", 
        mime='text/csv', 
        help="Click to download the SARS-CoV sequence dataset.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # MERS-CoV Dataset
    st.markdown("<div class='dataset-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='dataset-header'>MERS-CoV Spike Protein Sequences</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Middle East Respiratory Syndrome Coronavirus, first identified in 2012.</p>", unsafe_allow_html=True)
    
    df2 = pd.read_csv("S-human MERS-CoV.csv")
    st.dataframe(df2.style.set_properties(**{
        'background-color': '#f8f9fa', 
        'color': '#2C3E50',
        'border': '1px solid #eaeaea',
        'font-family': 'Segoe UI, Arial, sans-serif'
    }).highlight_max(axis=0, color='#E1F5FE'))
    
    csv2 = df2.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download MERS-CoV Dataset", 
        data=csv2, 
        file_name="MERS-CoV.csv", 
        mime='text/csv', 
        help="Click to download the MERS-CoV sequence dataset.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # SARS-CoV-2 Dataset
    st.markdown("<div class='dataset-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='dataset-header'>SARS-CoV-2 Spike Protein Sequences</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Novel Coronavirus responsible for COVID-19, first identified in 2019.</p>", unsafe_allow_html=True)
    
    df3 = pd.read_csv("S-human SARS-CoV-2.csv")
    st.dataframe(df3.style.set_properties(**{
        'background-color': '#f8f9fa', 
        'color': '#2C3E50',
        'border': '1px solid #eaeaea',
        'font-family': 'Segoe UI, Arial, sans-serif'
    }).highlight_max(axis=0, color='#E1F5FE'))
    
    csv3 = df3.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download SARS-CoV-2 Dataset", 
        data=csv3, 
        file_name="SARS-CoV-2.csv", 
        mime='text/csv', 
        help="Click to download the SARS-CoV-2 sequence dataset.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Vaccine & Drug Target Dataset
    st.markdown("<div class='dataset-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='dataset-header'>Vaccine & Drug Target Dataset</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Potential vaccine and drug targets for coronavirus treatments, including protein binding sites and epitopes.</p>", unsafe_allow_html=True)
    
    df4 = pd.read_csv("VACCINE & DRUG TARGET DATASET.csv")
    st.dataframe(df4.style.set_properties(**{
        'background-color': '#f8f9fa', 
        'color': '#2C3E50',
        'border': '1px solid #eaeaea',
        'font-family': 'Segoe UI, Arial, sans-serif'
    }).highlight_max(axis=0, color='#E1F5FE'))
    
    csv4 = df4.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Vaccine & Drug Target Dataset", 
        data=csv4, 
        file_name="VACCINE_DRUG_TARGET_DATASET.csv", 
        mime='text/csv', 
        help="Click to download the vaccine and drug target dataset.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 30px; color: #7F8C8D; font-size: 12px;'>
        These datasets are provided for research and educational purposes.
    </div>
    """, unsafe_allow_html=True)








# CONTACT US
if selected1=="Contact Us":
    with st.sidebar:
        selected3 = option_menu("Contact Us", ["Developers", 'Location'], 
    icons=['person','geo-alt'], 
    menu_icon="telephone", default_index=0, 
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "17px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "black"},
    }
)

    if selected3=='Developers':
              st.title('Developers')
              st.write('Jisha R C - jisha@gmail.com')   
              st.write('Achu Pushpan - achu@gmail.com')
              

    if selected3=='Location':
              st.title('Location')
              st.write('Amrita Viswa Vidyapeetham')
              st.write('Amrita School of Engineering')
              st.write('Amritapuri')
              st.write('Kollam')  







# INTRODUCTION
if selected1=="Introduction":
    selected = option_menu("PPINA", ['PROTEIN-PROTEIN INTERACTION NETWORK ANALYZER'], 
    icons=[''], 
    menu_icon="bezier", default_index=0, 
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
)
    original_title = '<p style="font-family:Arial; color:#37333F; font-size: 18px;"><b>PPINA is a web application tool which can be used for the construction of protein-protein interaction networks, its visualization and further analysis. The data for the protein-protein interaction network is collected from the String database. The tool also allows the user to create their own network by uploading the edge list. It provides a variety of features to analyze the networks for gaining biological and functional insights into the network.</b></p>'
    st.markdown(original_title, unsafe_allow_html=True)
    selected = option_menu(None, ['Sample Visualization of PPI Network'], 
    icons=['bezier'], 
    menu_icon="", default_index=0, 
    styles={
        "container": {"padding": "0!important", "background-color": "black"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "20px","text-color":"black", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "black"},
    }
)

    st.image('ae.png')

selected4='b'

# QUERY A SINGLE PROTEIN - FIXED VERSION
if selected2 == 'Query a Single Protein':
    selected = option_menu(
        "PPI NETWORK CONSTRUCTION", 
        ['Query a Single Protein'], 
        icons=['patch-question'], 
        menu_icon="bezier", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "15px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )
    
    p1 = st.text_input('Protein Name : ', 'ACE2')
    p2 = st.text_input('Organism NCBI ID : ', '9606')
    
    selected4 = option_menu(
        None, 
        ["Get PPI Data", "Visualize Network", "Statistical Analysis", 'Topological Analysis'], 
        icons=['file-bar-graph', 'bezier', "graph-up", 'graph-up'], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "15px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "black"},
        }
    )

    # GET PPI DATA
    if selected4 == "Get PPI Data":
        try:
            url = f'https://string-db.org/api/tsv/network?identifiers={p1}&species={p2}'
            r = requests.get(url, timeout=10)
            
            if r.status_code == 200:
                lines = r.text.split('\n')
                data = [l.split('\t') for l in lines]
                
                if len(data) > 1:
                    df = pd.DataFrame(data[1:-1], columns=data[0])
                    interactions = df[['preferredName_A', 'preferredName_B', 'score']]
                    data1 = pd.DataFrame()
                    data1['Source'] = interactions['preferredName_A']
                    data1['Target'] = interactions['preferredName_B']
                    
                    st.title(f'{p1} Protein-Protein Interaction')
                    st.dataframe(data1)
                    
                    csv = convert_df(data1)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='ppi_data.csv',
                        mime='text/csv',
                    )
                else:
                    st.error("No data returned from STRING database. Please check the protein name and NCBI ID.")
            else:
                st.error(f"Error connecting to STRING database. Status code: {r.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # VISUALIZE NETWORK
    if selected4 == 'Visualize Network':
        try:
            selected5 = option_menu(
                None, 
                ["Customize Nodes", "Customize Edges", "View Full Screen", "Exit Full Screen"], 
                icons=['pencil', 'pencil', "fullscreen", 'fullscreen-exit'], 
                menu_icon="cast", 
                default_index=3, 
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "15px"}, 
                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "green"},
                }
            )
            
            url = f'https://string-db.org/api/tsv/network?identifiers={p1}&species={p2}'
            r = requests.get(url, timeout=10)
            
            if r.status_code == 200:
                lines = r.text.split('\n')
                data = [l.split('\t') for l in lines]
                
                if len(data) > 1:
                    df = pd.DataFrame(data[1:-1], columns=data[0])
                    interactions = df[['preferredName_A', 'preferredName_B', 'score']]
                    data1 = pd.DataFrame()
                    data1['Source'] = interactions['preferredName_A']
                    data1['Target'] = interactions['preferredName_B']
                    
                    G = nx.from_pandas_edgelist(data1, 'Source', 'Target')
                    
                    a = '50%'
                    if selected5 == "Exit Full Screen":
                        a = '50%'
                    elif selected5 == "View Full Screen":
                        a = '100%'
                    
                    prot_net = Network(height='600px', width=a, bgcolor='#222222', font_color='white')
                    prot_net.from_nx(G)
                    prot_net.repulsion(
                        node_distance=420, 
                        central_gravity=0.33,
                        spring_length=110, 
                        spring_strength=0.10,
                        damping=0.95
                    )
                    
                    if selected5 == "Customize Nodes":
                        prot_net.show_buttons(filter_=['nodes'])
                        a = '50%'
                    elif selected5 == "Customize Edges":
                        prot_net.show_buttons(filter_=['edges'])
                        a = '50%'
                    
                    try:
                        path = '/tmp'
                        prot_net.save_graph(f'{path}/pyvis_graph.html')
                        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
                    except:
                        path = 'html_files'
                        os.makedirs(path, exist_ok=True)
                        prot_net.save_graph(f'{path}/pyvis_fil.html')
                        HtmlFile = open(f'{path}/pyvis_fil.html', 'r', encoding='utf-8')
                    
                    components.html(HtmlFile.read(), height=600)
                    
                    c1 = nx.number_of_nodes(G)
                    c2 = nx.number_of_edges(G)
                    
                    st.subheader("Network Overview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Proteins", c1)
                    with col2:
                        st.metric("Interactions", c2)
                        
                else:
                    st.error("No data available for visualization.")
            else:
                st.error(f"Error connecting to STRING database. Status code: {r.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # STATISTICAL ANALYSIS - FIXED VERSION
    if selected4 == 'Statistical Analysis':
        try:
            url = f'https://string-db.org/api/tsv/network?identifiers={p1}&species={p2}'
            r = requests.get(url, timeout=10)
            
            if r.status_code == 200:
                lines = r.text.split('\n')
                data = [l.split('\t') for l in lines]
                
                if len(data) > 1:
                    df = pd.DataFrame(data[1:-1], columns=data[0])
                    interactions = df[['preferredName_A', 'preferredName_B', 'score']]
                    data1 = pd.DataFrame()
                    data1['Source'] = interactions['preferredName_A']
                    data1['Target'] = interactions['preferredName_B']
                    
                    G = nx.from_pandas_edgelist(data1, 'Source', 'Target')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected7 = option_menu(
                            "STATISTICAL PARAMETERS", 
                            [
                                'Number of Nodes', 'Number of Edges', 'Average Degree', 
                                'Average Path Length', "Network Diameter", "Network Radius",
                                "Average Clustering Coefficient", 'Clustering Coefficient of each node',
                                "Connected or Disconnected", "Number of Connected Components",
                                "Center", "View Full Statistics"
                            ], 
                            icons=['123']*12,
                            menu_icon="graph-up", 
                            default_index=0,
                            styles={
                                "container": {"padding": "0!important", "background-color": "#fafafa", 'border': '7px solid powderblue'},
                                "icon": {"color": "orange", "font-size": "15px"}, 
                                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "green"},
                            }
                        )
                    
                    with col2:
                        if selected7 == "Number of Nodes":
                            c1 = nx.number_of_nodes(G)
                            st.header(f'Number of Nodes: {c1}')
                        
                        elif selected7 == "Number of Edges":
                            c2 = nx.number_of_edges(G)
                            st.header(f'Number of Edges: {c2}')
                        
                        elif selected7 == "Average Degree":
                            try:
                                degrees = [deg for node, deg in G.degree()]
                                avg_deg = sum(degrees) / len(degrees)
                                st.header(f'Average Degree: {avg_deg:.2f}')
                            except:
                                st.header('Average Degree cannot be calculated')
                        
                        elif selected7 == "Average Path Length":
                            try:
                                if nx.is_connected(G):
                                    c3 = nx.average_shortest_path_length(G)
                                    st.header(f'Average Path Length: {c3:.2f}')
                                else:
                                    st.header('Average Path Length cannot be calculated for disconnected graph')
                            except:
                                st.header('Average Path Length cannot be calculated')
                        
                        elif selected7 == "Network Diameter":
                            try:
                                if nx.is_connected(G):
                                    c4 = nx.diameter(G)
                                    st.header(f'Network Diameter: {c4}')
                                else:
                                    st.header('Network Diameter cannot be calculated for disconnected graph')
                            except:
                                st.header('Network Diameter cannot be calculated')
                        
                        elif selected7 == "Network Radius":
                            try:
                                if nx.is_connected(G):
                                    c5 = nx.radius(G)
                                    st.header(f'Network Radius: {c5}')
                                else:
                                    st.header('Network Radius cannot be calculated for disconnected graph')
                            except:
                                st.header('Network Radius cannot be calculated')
                        
                        elif selected7 == "Average Clustering Coefficient":
                            try:
                                G_cluster = list(nx.clustering(G).values())
                                if len(G_cluster) > 0:
                                    avg_clu = sum(G_cluster) / len(G_cluster)
                                    st.header(f'Average Clustering Coefficient: {avg_clu:.4f}')
                                else:
                                    st.header('No clustering coefficient data available')
                            except:
                                st.header('Average Clustering Coefficient cannot be calculated')
                        
                        elif selected7 == "Clustering Coefficient of each node":
                            try:
                                c6 = nx.clustering(G)
                                new = pd.DataFrame()
                                new['Nodes'] = list(c6.keys())
                                new['Clustering Coefficient'] = list(c6.values())
                                st.header('Clustering Coefficient of each node:')
                                st.dataframe(new)
                            except:
                                st.header('Clustering Coefficient cannot be calculated for each node')
                        
                        elif selected7 == "Connected or Disconnected":
                            try:
                                c7 = nx.is_connected(G)
                                status = "Connected" if c7 else "Disconnected"
                                st.header(f'The Network is {status}')
                            except:
                                st.header('Connectivity cannot be determined')
                        
                        elif selected7 == "Number of Connected Components":
                            try:
                                c8 = nx.number_connected_components(G)
                                st.header(f'Number of Connected Components: {c8}')
                            except:
                                st.header('Number of Connected Components cannot be calculated')
                        
                        elif selected7 == "Center":
                            try:
                                if nx.is_connected(G):
                                    c9 = nx.center(G)
                                    st.header(f'Center: {c9}')
                                else:
                                    st.header('Center cannot be calculated for disconnected graph')
                            except:
                                st.header('Center cannot be found')
                        
                        elif selected7 == "View Full Statistics":
                            try:
                                stats_data = []
                                
                                # Number of Nodes
                                try:
                                    nodes = nx.number_of_nodes(G)
                                    stats_data.append(("Number of Nodes", nodes))
                                except:
                                    stats_data.append(("Number of Nodes", "N/A"))
                                
                                # Number of Edges
                                try:
                                    edges = nx.number_of_edges(G)
                                    stats_data.append(("Number of Edges", edges))
                                except:
                                    stats_data.append(("Number of Edges", "N/A"))
                                
                                # Average Degree
                                try:
                                    degrees = [deg for node, deg in G.degree()]
                                    avg_deg = sum(degrees) / len(degrees) if degrees else 0
                                    stats_data.append(("Average Degree", f"{avg_deg:.2f}"))
                                except:
                                    stats_data.append(("Average Degree", "N/A"))
                                
                                # Average Path Length
                                try:
                                    if nx.is_connected(G):
                                        avg_path = nx.average_shortest_path_length(G)
                                        stats_data.append(("Average Path Length", f"{avg_path:.2f}"))
                                    else:
                                        stats_data.append(("Average Path Length", "Disconnected graph"))
                                except:
                                    stats_data.append(("Average Path Length", "N/A"))
                                
                                # Network Diameter
                                try:
                                    if nx.is_connected(G):
                                        diameter = nx.diameter(G)
                                        stats_data.append(("Network Diameter", diameter))
                                    else:
                                        stats_data.append(("Network Diameter", "Disconnected graph"))
                                except:
                                    stats_data.append(("Network Diameter", "N/A"))
                                
                                # Network Radius
                                try:
                                    if nx.is_connected(G):
                                        radius = nx.radius(G)
                                        stats_data.append(("Network Radius", radius))
                                    else:
                                        stats_data.append(("Network Radius", "Disconnected graph"))
                                except:
                                    stats_data.append(("Network Radius", "N/A"))
                                
                                # Average Clustering Coefficient
                                try:
                                    cluster_values = list(nx.clustering(G).values())
                                    avg_cluster = sum(cluster_values) / len(cluster_values) if cluster_values else 0
                                    stats_data.append(("Average Clustering Coefficient", f"{avg_cluster:.4f}"))
                                except:
                                    stats_data.append(("Average Clustering Coefficient", "N/A"))
                                
                                # Number of Connected Components
                                try:
                                    components = nx.number_connected_components(G)
                                    stats_data.append(("Number of Connected Components", components))
                                except:
                                    stats_data.append(("Number of Connected Components", "N/A"))
                                
                                # Create DataFrame
                                df_stats = pd.DataFrame(stats_data, columns=["Statistical Parameter", "Value"])
                                st.header('Network Statistics')
                                st.dataframe(df_stats, hide_index=True)
                                
                            except Exception as e:
                                st.error(f"Error calculating statistics: {str(e)}")
                
                else:
                    st.error("No data available for analysis.")
                    
            else:
                st.error(f"Error connecting to STRING database. Status code: {r.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
