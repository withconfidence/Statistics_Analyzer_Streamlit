# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:54:36 2021

@author: Robert https://github.com/rdzudzar
"""
# Package imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np
#from scipy import stats
import scipy.stats
from scipy.stats import wasserstein_distance, energy_distance,  ks_2samp
import math
import json

from bokeh.plotting import figure
from bokeh.models import Legend
#from bokeh.io import curdoc

# Helper function imports
# These are pre-computed so that they don't slow down the App
from helper_functions import distr_selectbox_names, creating_dictionaries, calculate_psi

import time
import base64
import collections


def page_comparison():
    """
    The fit page in this app made with Streamlit is for a fitting a selected
    distribution(s) to the User imported data.
    """
    name_docstring_dict, name_eq_dict, name_proper_dict, \
        all_dist_params_dict, name_url_dict = creating_dictionaries()
    
    st.sidebar.info("""
                Import your data as a **.csv** file 
                and follow instructions to fit a
                continuous distribution(s) to your data.
                """)
                
    # Add a bit empy space before showing About
    st.sidebar.text("")

    st.sidebar.markdown("**Select Figure Mode:**")
    plot_mode = st.sidebar.radio("Options", ('Dark Mode', 'Light Mode'))   

    st.sidebar.text("")
    st.sidebar.text("")

    st.markdown("<h1 style='text-align: center;'> Fit distribution(s) </h1>", 
                unsafe_allow_html=True)

    
    #Streamlit Sharing if you set the config option in .streamlit/config.toml:
    #[server]
    #maxUploadSize=2
    
    # # Using cache as we perform only once 'loading' of the data
    # @st.cache
    # def load_csv():
    #     """ Get the loaded .csv into Pandas dataframe. """
    #
    #     df_load_1 = pd.read_csv(input_1, sep=None , engine='python',
    #                          encoding='utf-8')
    #
    #     return df_load_1
    #
    # # Using cache as we perform only once 'loading' of the data
    # @st.cache
    # def load_csv_2():
    #     """ Get the loaded .csv into Pandas dataframe. """
    #
    #     df_load_2 = pd.read_csv(input_2, sep=None, engine='python',
    #                             encoding='utf-8')
    #
    #     return df_load_2
   
    # Streamlit - upload files
    input_1 = st.file_uploader('Choose first CSV file')
    input_2 = st.file_uploader('Choose second CSV file')
    
    # Ask for upload if not yet.
    if input_1 is None or input_2 is None:
        st.write('Upload your data, or:')
        download_sample = st.checkbox("Download sample data")
    
    # Get the Sample file from App GitHub repository
    try:
        if download_sample:
            st.markdown(""" [Download](https://github.com/rdzudzar/DistributionAnalyser/blob/main/sample_data/sample_da.csv)""")            
            st.markdown("""**after you download data, 
                        upload them above.*""")

    except:
        # If the user imports file - parse it, and ask User to select a column.
        if input_1 and input_2:
            with st.spinner('Loading data...'):
                # Pass to a function above so we can use st.cache
                df1 = pd.read_csv(input_1, sep=None, engine='python', encoding='utf-8')
                columns1 = list(df1.columns)
                df2 = pd.read_csv(input_2, sep=None, engine='python', encoding='utf-8')
                columns2 = list(df2.columns)
                columns = [x + " : " + y for x, y in zip(columns1, columns2)]
    
                # User can see which columns are present in the imported file
                st.write("Available columns (expand to see):")
                st.write(columns)
                
                # Select menu for User to pick the data
                # This 'data_col' selection is parsed as selected dataframe
                data_col = st.selectbox("Select input",
                                        options=list(range(len(columns))),
                                        format_func=lambda x: columns[x])
                # Get the selected column
                # Replace inf/-inf with NaN and remove NaN if present
                key_col1 = df1.columns[data_col]
                df1 = df1[key_col1].replace([np.inf, -np.inf], np.nan).dropna()
                key_col2 = df2.columns[data_col]
                df2 = df2[key_col2].replace([np.inf, -np.inf], np.nan).dropna()
            
    
    def plot(df, data_stat):
        """ 
        Histogram of the input data. Contains also information about the 
        Figure style, depending on the active Mode.
        """
        
        if plot_mode == 'Light Mode':
            hist_edge_color = 'black'
            hist_color= 'white'
            quant_color = 'black'
            median_color = 'black'
            pdf_color = '#08519c'
            cdf_color = 'black'
            plt.style.use('classic')
            plt.rcParams['figure.facecolor'] = 'white'


        if plot_mode == 'Dark Mode':
            hist_edge_color = 'black'
            hist_color= 'white'
            median_color = 'magenta'
            quant_color = 'white'
            pdf_color = '#fec44f'
            cdf_color = 'white'
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = 'black'

        fig, ax = plt.subplots(1,1)
        
        # Plot hist
        ax.hist(df, bins=round(math.sqrt(len(df))), 
                density=True, color=hist_color, 
                ec=hist_edge_color, alpha=0.3)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tick_params(top=False, bottom=True, left=True, right=False,
                labelleft=True, labelbottom=True)

        ax.set_xlabel(f'{data_col}')
        ax.set_ylabel('Density')
        
        # If user selects data_stat
        if data_stat:
            # Hist contains tuple: n bins, (n+1) bin boundaries
            hist = np.histogram(df, bins=round(math.sqrt(len(df))))
            #Generates a distribution given by a histogram.
            hist_dist = scipy.stats.rv_histogram(hist)
            x_plot = np.linspace(min(df), max(df), len(df))

    
            q = [0.05, 0.25, 0.50, 0.75, 0.95]
            n = ['5th','25th','50th','75th','95th']
            quantiles = df.quantile(q)
            q_max = hist_dist.cdf(quantiles)

            
            for i, qu in enumerate(quantiles):
                ax.plot(qu, q_max[i], alpha=0.5, color=quant_color,
                        markersize=10, marker='D')
                ax.text(qu, q_max[i]+(q_max[i]/10), f'{n[i]}', ha='center',
                        alpha=0.5)
            ax.scatter([], [], alpha=0.5, color=quant_color, marker='D', 
                       label='Percentiles')
            # The pdf is defined as a stepwise function from the provided histogram.
            # The cdf is a linear interpolation of the pdf.
            ax.plot(x_plot, hist_dist.pdf(x_plot), linewidth=2,
                    color=pdf_color, label='PDF')
            ax.plot(x_plot, hist_dist.cdf(x_plot), linewidth=2,
                    color=cdf_color, label='CDF')
            
            
            ax.vlines(np.mean(df), ymin=0, ymax=hist_dist.cdf(np.mean(df)),
                      color='red', linestyle='--', linewidth=2,
                      label=f'Mean {round(np.mean(df),2)}')
            ax.vlines(np.median(df), ymin=0, ymax=hist_dist.cdf(np.median(df)),
                      color=median_color, linestyle='--', linewidth=2,
                      label=f'Median {round(np.median(df),2)}')            
       
            
            leg = plt.legend(loc=0)
            leg.get_frame().set_edgecolor("#525252")

        return fig
    
    def bokeh_set_plot_properties(plot_mode, n):
        """
        Constructs a list of properties that will be assigned to a Bokeh
        figure depending whether it is in the Light or Dark Mode.

        Parameters
        ----------
        plot_mode : string; plot 'Dark Mode' or 'Light Mode'

        Returns
        -------
        p : Bokeh Figure
        colors_cmr : Colors from the colormap to be assigned to lines

        """
                
        p = figure(height=450, width=700)
        
        p.add_layout(Legend(), 'right')
        p.legend.title = '15 Best Fits and their SSE'
        p.legend.background_fill_alpha = 1
        p.legend.label_text_font_size = '11pt'
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.xaxis.axis_label = f'{data_col}'
        p.yaxis.axis_label = 'Density' 


        if plot_mode == 'Dark Mode':
            text_color = 'white'
            back_color = 'black'
            legend_color = 'yellow'
            
            # It will get n colors from cmasher rainforest map
            # if n>15, it will take 15; otherwise n will be the 
            # lengthe of the chosed distributions (n defined line 685)
            colors_cmr = cmr.take_cmap_colors('cmr.rainforest_r', 
                                      n, cmap_range=(0.1, 0.7), 
                                     return_fmt='hex')     
        
        if plot_mode == 'Light Mode':
            text_color = 'black'
            back_color = 'white'
            legend_color = 'blue'   
        
            colors_cmr = cmr.take_cmap_colors('cmr.rainforest', 
                                      n, cmap_range=(0.2, 0.9), 
                                     return_fmt='hex')
        
        p.legend.title_text_color = text_color
        p.yaxis.major_label_text_color = text_color
        p.xaxis.axis_label_text_color = text_color
        p.xaxis.major_label_text_color = text_color
        p.yaxis.axis_label_text_color = text_color
        p.xaxis.major_tick_line_color = text_color
        p.yaxis.major_tick_line_color = text_color
        p.xaxis.minor_tick_line_color = text_color
        p.yaxis.minor_tick_line_color = text_color
        p.xaxis.axis_line_color = text_color
        p.yaxis.axis_line_color = text_color
        
        p.border_fill_color = back_color
        p.background_fill_color = back_color
        p.legend.background_fill_color = back_color
        p.legend.label_text_color = legend_color
        p.title.text_color = legend_color
        p.outline_line_color = back_color
        
        return p, colors_cmr
       
      
    def bokeh_pdf_plot_results(df, results, n):
        """
        Process results and plot them on the Bokeh Figure. User can interact
        with the legend (clicking on the items will enhance lines on Figure)

        Parameters
        ----------
        df : input data
        results : nested list (contains tuples) with the data from the 
                fitting (contains [sse, arg, loc, scale])
        n : integer; First n best fit PDFs to show on the Figure.
        plot_mode : string; 'Dark Mode' or 'Light Mode' (connected with radio
                                                         button)

        Returns
        -------
        p : Returns Bokeh interactive figure (data histogram+best fit PDFs)

        """
                 
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        hist, edges = np.histogram(df, density=True, 
                                   bins=round(math.sqrt(len(df))))
        
    
        # Obtain Figure mode from the function:  bokeh_set_plot_properties
        p, colors_cmr = bokeh_set_plot_properties(plot_mode, n)
        
        # Bokeh histogram
        p.quad(top=hist, bottom=0, left=edges[:-1], 
               right=edges[1:], line_color="black",
               line_width = 0.3,
               fill_color='white', fill_alpha = 0.3)
        
        # Plot each fitted distribution
        i = -1
        for distribution, result in fit_dict_res.items():
            i += 1
            
            sse = round(result[0],2)
            arg = result[1]
            loc = result[2]
            scale = result[3]

            best_params = result[1:4] 
            flat_list = list(flatten(best_params)) 
            param_names = (distribution.shapes + ', loc, scale').split(', ') if distribution.shapes else ['loc', 'scale']
            param_str = ', '.join([f'{k} = {round(v,2)}' for k,v 
                                   in zip(param_names, flat_list)])

            # Generate evenly spaced numbers over a specified interval
            # Make pdf/cdf with the parameters of fitted functions
            x_plot = np.linspace(min(df), max(df), 400)
            y_plot = distribution.pdf(x_plot, loc=loc, scale=scale, *arg)

            # The best fit distribution will be with i=0
            if i == 0:
                # Bokeh line plot with interactive legend
                line = p.line(x_plot, y_plot, line_width=5,
                       line_color = colors_cmr[0],
                       legend_label=str(distribution.name) + ": " + str(sse)
                       )
                line.visible = True
                p.legend.click_policy = "hide"
                p.title.text = f'Best fit {distribution.name}: {param_str}'
                
                
                if distribution.name in name_eq_dict.keys():
                        
                    st.markdown(f"""You can read more about best fit distribution:                        
                        [**{name_url_dict[distribution.name][1]}**]
                            ({name_url_dict[distribution.name][0]})
                        """)

            # Next 15 best fits; 15 is arbitrary taken.
            elif (i>0) and (i < 15):
                lines = p.line(x_plot, y_plot, line_width=2.5,
                                line_dash="10 2",
                       line_color = colors_cmr[i],
                       legend_label =str(distribution.name) + ": " + str(sse)
                        )
                lines.visible = False
                p.legend.click_policy = "hide"

            else:
                pass
                   
        return p  


    def bokeh_cdf_plot_results(df, results, n):
        """
        Process results and plot them on the Bokeh Figure. User can interact
        with the legend (clicking on the items will enhance lines on Figure)

        Parameters
        ----------
        df : input data
        results : nested list (contains tuples) with the data from the 
                fitting (contains [sse, arg, loc, scale])
        n : integer; First n best fit CDFs to show on the Figure.
        plot_mode : string; 'Dark Mode' or 'Light Mode' (connected with radio
                                                         button)

        Returns
        -------
        p : Returns Bokeh interactive figure (data hostogram+best fit CDFs)

        """
        
        # Hist contains tuple: n bins, (n+1) bin boundaries
        hist_data = np.histogram(df, bins=round(math.sqrt(len(df))))
        #Generates a distribution given by a histogram.
        hist_dist_data = scipy.stats.rv_histogram(hist_data)
        x_plot_data = np.linspace(min(df), max(df), 400)
          
    
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        
        hist, edges = np.histogram(df, density=True, 
                                   bins=round(math.sqrt(len(df))))
        
    
        # Obtain Figure mode from the function:  bokeh_set_plot_properties
        p, colors_cmr = bokeh_set_plot_properties(plot_mode, n)
        
        # Bokeh histogram
        p.quad(top=hist, bottom=0, left=edges[:-1], 
               right=edges[1:], line_color="black",
               line_width = 0.3,
               fill_color='white', fill_alpha = 0.3)
        
        p.line(x_plot_data, hist_dist_data.cdf(x_plot_data), 
                          line_color='red', legend_label='CDF sample data',
                          line_width=3)  
        p.legend.click_policy = "hide"

        # Plot each fitted distribution
        i = -1
        for distribution, result in fit_dict_res.items():
            i += 1
            
            sse = round(result[0],2)
            arg = result[1]
            loc = result[2]
            scale = result[3]

            best_params = result[1:4] 
            flat_list = list(flatten(best_params)) 
            param_names = (distribution.shapes + ', loc, scale').split(', ') if distribution.shapes else ['loc', 'scale']
            param_str = ', '.join([f'{k} = {round(v,2)}' for k,v 
                                   in zip(param_names, flat_list)])

            # Generate evenly spaced numbers over a specified interval
            # Make pdf/cdf with the parameters of fitted functions
            x_plot = np.linspace(min(df), max(df), len(df))

            y_plot = distribution.cdf(x_plot, loc=loc, scale=scale, *arg)
                
                
            # The best fit distribution will be with i=0
            if i == 0:
                # Bokeh line plot with interactive legend
                line = p.line(x_plot, y_plot, line_width=5,
                       line_color = colors_cmr[0],
                       legend_label=str(distribution.name) + ": " + str(sse)
                       )
                line.visible = True
                p.legend.click_policy = "hide"
                p.title.text = f'Best fit {distribution.name}: {param_str}'

            # Next 15 best fits; 15 is arbitrary taken.
            elif (i>0) and (i < 15):
                lines = p.line(x_plot, y_plot, line_width=2.5,
                                line_dash="10 2",
                       line_color = colors_cmr[i],
                       legend_label =str(distribution.name) + ": " + str(sse)
                        )
                lines.visible = False
                p.legend.click_policy = "hide"

            else:
                pass
                   
        return p

    
    @st.cache(allow_output_mutation=True)
    def fit_data(df):
        """ 
        Modified from: https://stackoverflow.com/questions/6620471/fitting\
            -empirical-distribution-to-theoretical-ones-with-scipy-python 
        
        This function is performed with @cache - storing results in the local
        cache; read more: https://docs.streamlit.io/en/stable/caching.html
        """
        
        # If the distribution(s) are selected in the selectbox
        if chosen_distr:
            
            # Check for nan/inf and remove them
            ## Get histogram of the data and histogram parameters
            num_bins = round(math.sqrt(len(df)))
            hist, bin_edges = np.histogram(df, num_bins, density=True)
            central_values = np.diff(bin_edges)*0.5 + bin_edges[:-1]

            results = {}
            for distribution in chosen_distr:
                
                # Go through distributions
                dist = getattr(scipy.stats, distribution)
                # Get distribution fitted parameters
                params = dist.fit(df)
                
                ## Separate parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
            
                ## Obtain PDFs
                pdf_values = [dist.pdf(c, loc=loc, scale=scale, *arg) for c in
                              central_values]
        
                # Calculate the RSS: residual sum of squares 
                # Also known as SSE: sum of squared estimate of errors
                # The sum of the squared differences between each observation\
                # and its group's mean; here: diff between distr. & data hist
                sse = np.sum(np.power(hist - pdf_values, 2.0))
                
                
                # Parse fit results 
                results[dist] = [sse, arg, loc, scale]

            # containes e.g. [0.5, (13,), 8, 1]
            # equivalent to  [sse, (arg,), loc, scale] as argument number varies
            results = {k: results[k] for k in sorted(results, key=results.get)}

            return results

    def flatten(nested_l):
        
        """
        Flatten the list and take care if there are tuples in the list.
        Arguments can be multiple (a, b, c...). This function is from:
        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
        """
        if isinstance(nested_l, collections.Iterable):
            return [a for i in nested_l for a in flatten(i)]
        else:
            return [nested_l]
        

    def results_to_dataframe(df, results):
        """ 
        This function takes the results from the fitting and parses it to 
        produce variables that will be storred into PandasDataframe for
        the easy overview.
        """
        
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        
        df_dist = []
        df_params = []
        df_sse = []
        for distribution, result in fit_dict_res.items():
            sse = result[0]
            best_params = result[1:4] 

            flat_list = list(flatten(best_params))
            
            param_names = (distribution.shapes + ',loc,scale').split(',') if distribution.shapes else ['loc', 'scale']
            param_str = ', '.join([f'{k} = {round(v,2)}' for k,v 
                                   in zip(param_names, flat_list)])
            
            #title = f'{distribution.name}: {param_str}'
            
            df_dist.append(f'{distribution.name}')
            df_params.append(f'{param_str}')
            df_sse.append(round(sse, 4))

        fit_results = pd.DataFrame(
                {'Distribution': df_dist,
                 'Fit Parameters': df_params,
                 'SSE': df_sse}
                )

        return fit_results 
    
    def produce_output_for_code_download_parameters(df, results):
        """
        Parse the best fit function and parameters to generate python
        code for User. Works fine for all current forms of the 
        continuous functions (with various numbers of shape parameters).
        """
        
        # Need to start loop as I want to separate first item
        i = -1
        # Pasing dictionary with best fit results
        fit_dict_res = fit_data(df)
        for distribution, result in fit_dict_res.items():
            i += 1
            if i == 0:
                # Need to add to to check if there are shapes or not
                if distribution.shapes is not None:
                    shapes = distribution.shapes+str(',')
                else:
                    shapes = ""
                #print(shapes)
            else:
                pass
        df_results = results_to_dataframe(df, results)
        
        best_dist = df_results['Distribution'][0]
        
        fit_params_all = df_results['Fit Parameters']
          
        # Get scale
        best_scale = fit_params_all[0].split(", ")[-1]
        # Get loc 
        best_loc = fit_params_all[0].split(", ")[-2]
        # Get all arguments
        args = fit_params_all[0].split(",")[0:-2]
        # String manipulation to matches desired form for the code generation
        args = str([i for i in args]).strip(" [] ").strip("'").replace("'", '').replace(" ", '').replace(",",'\n')

        return shapes, best_dist, best_scale, best_loc, args, fit_params_all[0]
    
    def get_code():
        """ Prints out the python formatted code"""
        
        st.code(f"{generate_fit_code}")    

    def result_downloader(_text_):
        """
        Strings <-> bytes conversions and creating a link which will
        download generated python script.
        """

        # Add a timestamp to the name of the saved file
        time_stamp = time.strftime("%Y%m%d_%H%M%S")

        # Base64 takes care of porting info to the data
        b64 = base64.b64encode(_text_.encode()).decode()
        
        # Saved file will have distribution name and the timestamp
        code_file = f"{time_stamp}.json"
        st.markdown(f'** Export Result **: \
                    <a href="data:file/txt;base64,{b64}" \
                        download="{code_file}">Click Here</a>', 
                        unsafe_allow_html=True)
    
    def csv_downloader(data):
        """
        Strings <-> bytes conversions and creating a link which will
        download generated csv file with the DataFrame that contains fitting
        results.
        """
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        
        csvfile = data.to_csv()
        
        b64 = base64.b64encode(csvfile.encode()).decode()
        
        new_filename = f"fit_results{time_stamp}.csv"
        href = f'** Download DataFrame as .csv: ** \
            <a href="data:file/csv;base64,{b64}" \
            download="{new_filename}">Click Here</a>'
        st.markdown(href, unsafe_allow_html=True)

    def plot_comparison(df1, df2):
        """
        Histogram of the input data. Contains also information about the
        Figure style, depending on the active Mode.
        """

        if plot_mode == 'Light Mode':
            hist_edge_color = 'black'
            hist_color = 'white'
            quant_color = 'black'
            median_color = 'black'
            pdf_color = '#08519c'
            cdf_color = 'black'
            plt.style.use('classic')
            plt.rcParams['figure.facecolor'] = 'white'

        else:
            hist_edge_color = 'black'
            hist_color = 'white'
            median_color = 'magenta'
            quant_color = 'white'
            pdf_color = '#fec44f'
            cdf_color = 'white'
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = 'black'

        fig, ax = plt.subplots(1, 1)

        bins = round(math.sqrt(max(len(df1), len(df2))))
        # Plot hist
        # ax.hist([df1, df2], bins=bins)
                # density=True, color=hist_color,
                # ec=hist_edge_color, alpha=0.3)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tick_params(top=False, bottom=True, left=True, right=False,
                        labelleft=True, labelbottom=True)

        ax.set_xlabel(f'{data_col}')
        ax.set_ylabel('Density')

        x_plot = np.linspace(min(min(df1), min(df2)), max(max(df1), max(df2)), max(len(df1), len(df2)))
        q = [0.05, 0.25, 0.50, 0.75, 0.95]
        n = ['5th', '25th', '50th', '75th', '95th']

        # For First Data
        # Hist contains tuple: n bins, (n+1) bin boundaries
        hist1 = np.histogram(df1, bins=bins)
        # hist_2 = np.histogram(df2, bins=round(math.sqrt(len(df2))))
        # Generates a distribution given by a histogram.
        hist_dist1 = scipy.stats.rv_histogram(hist1)

        quantiles1 = df1.quantile(q)
        q_max1 = hist_dist1.cdf(quantiles1)
        for i, qu in enumerate(quantiles1):
            ax.plot(qu, q_max1[i], alpha=0.5, color="red",
                    markersize=10, marker='D')
            ax.text(qu, q_max1[i] + (q_max1[i] / 10), f'{n[i]}', ha='center',
                    color="red", alpha=0.5)

        # The pdf is defined as a stepwise function from the provided histogram.
        # The cdf is a linear interpolation of the pdf.
        ax.plot(x_plot, hist_dist1.cdf(x_plot), linewidth=2,
                color="red", label='First')

        ax.vlines(np.mean(df1), ymin=0, ymax=hist_dist1.cdf(np.mean(df1)),
                  color='red', linestyle='--', linewidth=2,
                  label=f'Mean {round(np.mean(df1), 2)}')
        ax.vlines(np.median(df1), ymin=0, ymax=hist_dist1.cdf(np.median(df1)),
                  color=median_color, linestyle='--', linewidth=2,
                  label=f'Median {round(np.median(df1), 2)}')


        # For Second Data
        # Hist contains tuple: n bins, (n+1) bin boundaries
        hist2 = np.histogram(df2, bins=bins)
        # Generates a distribution given by a histogram.
        hist_dist2 = scipy.stats.rv_histogram(hist2)
        quantiles2 = df2.quantile(q)
        q_max2 = hist_dist2.cdf(quantiles2)
        for i, qu in enumerate(quantiles2):
            ax.plot(qu, q_max2[i], alpha=0.5, color="blue",
                    markersize=10, marker='D')
            ax.text(qu, q_max2[i] + (q_max2[i] / 10), f'{n[i]}', ha='center',
                    color="blue", alpha=0.5)
        # The pdf is defined as a stepwise function from the provided histogram.
        # The cdf is a linear interpolation of the pdf.
        ax.plot(x_plot, hist_dist2.cdf(x_plot), linewidth=2,
                color="blue", label='Second')



        # The pdf is defined as a stepwise function from the provided histogram.
        # The cdf is a linear interpolation of the pdf.
        # ax.plot(x_plot, hist_dist.pdf(x_plot), linewidth=2,
        #         color=pdf_color, label='PDF')


        ax.vlines(np.mean(df2), ymin=0, ymax=hist_dist2.cdf(np.mean(df2)),
                  color='red', linestyle='--', linewidth=2,
                  label=f'Mean {round(np.mean(df2), 2)}')
        ax.vlines(np.median(df2), ymin=0, ymax=hist_dist2.cdf(np.median(df2)),
                  color=median_color, linestyle='--', linewidth=2,
                  label=f'Median {round(np.median(df2), 2)}')

        # ax.scatter([], [], alpha=0.5, color=quant_color, marker='D',
        #            label='Percentiles')

        leg = plt.legend(loc=0)
        leg.get_frame().set_edgecolor("#525252")

        return fig



    # Checks steps by steps to ensure the flow of the data input; examination,
    # fitting and the display of the results.
    if input_1 and input_2:
        

        # display_list = ["File1 Distribution", "File2 Distribution", "Compare Distribution"]
        # display_item = st.selectbox("Select the item to be displayed.", list(range(len(display_list))), format_func=lambda x:display_list[x])
        # if display_item == 2:
        #     st.write("please compare distribution")
        # else:
            # if display_item == 0:
            #     df = df1
            # else:
            #     df = df2
        st.write("Examine your data:")
        col1, col2 = st.columns(2)
        display_dist = col1.checkbox('Show Sample Distribution')
        compare_dist = col2.checkbox("Compare Distribution", value=False)
        df1_info = {
            "Max": max(df1),
            "Min": min(df1),
            "Mean": np.mean(df1),
            "Median": np.median(df1)}
        df2_info = {
            "Max": max(df2),
            "Min": min(df2),
            "Mean": np.mean(df2),
            "Median": np.median(df2)}
        if display_dist:
            st.write("File1 Distribution")
            st.pyplot(plot(df1, True))
            st.info(df1_info)
            st.write("File2 Distribution")
            st.pyplot(plot(df2, True))
            st.info(df2_info)
        if compare_dist:

            st.pyplot(plot_comparison(df1, df2))

            eval_labels = [
                "Kolmogorov–Smirnov (KS) test",
                "First Wassertein Distance (Earth Mover’s distance)",
                "Cramér - von Mises(CM) distance(Energy distance)",
                "Population StabilityIndex (PSI)"
            ]
            # display_item = st.multiselect("Select the items to be displayed.", list(range(len(eval_labels))),
            #                             format_func=lambda x: eval_labels[x])

            two_sided_stat, two_sided_p = ks_2samp(df1, df2, alternative="two_sided")
            less_stat, less_p = ks_2samp(df1, df2, alternative="less")
            greater_stat, greater_p = ks_2samp(df1, df2, alternative="greater")
            ks_stat = {
                "two_sided": {"stat": two_sided_stat, "p_value": two_sided_p},
                "less": {"stat": less_stat, "p_value": less_p},
                "greater": {"stat": greater_stat, "p_value": greater_p}
            }

            st.info("Kolmogorov–Smirnov Test")
            st.write("{}".format(ks_2samp(df1, df2, alternative="two_sided")))
            # st.table(ks_stat)
            # other_index = {
            #     # {"ks": ks_stat},
            #     "First Wasserstein Distance": [wasserstein_distance(u_values=df1, v_values=df2)],
            #     "Cramér - von Mises(CM) distance(Energy distance)": [energy_distance(df1, df2)],
            #     "Population Stability Index(PSI)": [calculate_psi(df1, df2)]
            # }
            # other_index_df = pd.DataFrame(other_index)
            st.info("First Wasserstein Distance")
            st.write(wasserstein_distance(u_values=df1, v_values=df2))
            st.info("Cramér - von Mises(CM) distance(Energy distance)")
            st.write(energy_distance(df1, df2))
            st.info("Population Stability Index(PSI)")
            st.write(calculate_psi(df1, df2))

            # st.table(other_index_df.assign(hack='').set_index('hack'))
            result = {
                "fist_file_distribution": df1_info,
                "second_file_distribution": df2_info,
                "KS_test": ks_2samp(df1, df2, alternative="two_sided"),
                "wasserstein_distance": wasserstein_distance(u_values=df1, v_values=df2),
                "cm_distance": energy_distance(df1, df2),
                "psi": calculate_psi(df1, df2)
            }
            result_downloader(json.dumps(result, indent=4))


    return