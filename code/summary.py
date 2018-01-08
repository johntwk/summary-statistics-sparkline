# Class Name: summary
# Date      : December 28th, 2017
# Purpose   : take a Pandas dataframe and print summary statistics, 
#             draw sparklines, generate latex code for reporting, and 
#             generate pdf report
class summary:
    df   = None
    mean = None
    var  = None
    sd   = None
    min  = None
    min_id = None
    max  = None
    max_id = None
    len  = None
    user_defined_val_name_lst = []
    user_defined_val_val_lst = []
    summary_df = None
            
    def __init__(self, df, axis = 0,                    # Dataframe function selection
                 user_defined_val=None, columns = None, # Dataframe user-defined functions
                 ddof = 1):              # degree of freedom
        # Import required libraries
        import pandas as pd
        import numpy as np
        import types
        
        def error_check_dataframe(df, axis):
            # Error Check: Dataframe
            if (not isinstance(df, pd.DataFrame)):
                raise ValueError('df must be a pandas dataframe, but it is',type(df))
            axis = int(axis)
            if (axis != 0 and axis != 1):
                raise ValueError('axis must be either 0 or 1, but entered',axis)
        def error_check_user_defined(u):
            if (u is not None):
                if (not isinstance(u,dict)):
                    raise ValueError('df must be a pandas dataframe, but it is',type(df))
                for key, val in u.items():
                    if (not(isinstance(key,str))):
                        msg = '''user_defined_val contains illegal entries.
                                 The key '''+str(key)+'is not a string'
                        raise ValueError(msg)
                    if (not isinstance(val,types.FunctionType)):
                        msg = '''Value of dictionary user_defined_val is illegal.
                               The value must be callable, but it is a '''+str(type(val))
                        raise ValueError(msg)
                    # Also need to check if the return type of the function a numerical value
        def error_check_columns(df,columns):
            if (not columns is None):
                col_lst = df.columns
                for col in columns:
                    if (not col in col_lst):
                        raise ValueError(col+' is not a column in the dataframe df.')
        def error_check_ddof(ddof):
            if (not isinstance(ddof,int)):
                raise ValueError('ddof must be an integer, but it is a'+str(type(ddof)))
            #if (not isinstance(decimal_place,int)):
            #    raise ValueError('decimal_place must be an integer, but it is a'+str(type(decimal_place)))
        def error_check_df_cols(df,columns):
            # check if every value of the df numeric or None or np.nan
            for col in columns:
                try:
                    df[col] = df[col].astype(float)
                except:
                    raise ValueError('Column '+col+'contains non-numerical values')
                    
        # Call error check functions    
        error_check_dataframe(df = df, axis=axis)
        error_check_user_defined(user_defined_val)
        error_check_columns(df=df,columns=columns)
        error_check_df_cols(df=df,columns=columns)
        error_check_ddof(ddof)
        
        # Filter columns
        if (columns is not None):
            self.df = df[columns]
            df = self.df
            col_lst = columns
        else:
            self.df = df
            col_lst = df.columns
        
        # Calculate descriptive statistics
        self.mean = df.mean(axis = axis)
        self.var  = df.var(axis = axis, ddof = ddof)
        self.sd   = (self.var)**(0.5)
        self.min  = df.min(axis = axis)
        self.min_id = df.idxmin(axis=axis)
        self.max  = df.max(axis = axis)
        self.max_id = df.idxmin(axis=axis)
        self.len = len(df)
        
        # Calculate user-defined statistics
        if (not user_defined_val is None):
            for key, func in user_defined_val.items():
                self.user_defined_val_name_lst.append(key)
                if (axis == 0):
                    for col in col_lst:
                        # Apply the functions
                        self.user_defined_val_val_lst.append(func(df[col]))
                else:
                    for index in range(0,self.len):
                        self.user_defined_val_val_lst.append(func(df.loc[index]))
        def produce_summary_df():
            sum_dict = {'Number of Observations':self.len,
                        'Mean':self.mean, 'Variance':self.var, 'Standard Deviation':self.sd,
                        'Minimum':self.min, 'Index of Minimum':self.min_id,
                        'Maximum':self.max, 'Index of Maximum':self.max_id
                       }
            if (len(self.user_defined_val_name_lst) != 0):
                #print 'List'
                for name, val in zip(self.user_defined_val_name_lst,self.user_defined_val_val_lst):
                    sum_dict[name] = val
            sum_df = pd.DataFrame.from_dict(sum_dict)
            sum_df = sum_df[sum_dict.keys()]
            return sum_df
        self.summary_df = produce_summary_df()
        
    def print_summary(self, ncols=4, width = 80, decimal_places=2):
        import tabulate
        import pandas as pd
        import numpy as np
        print "---------------------"
        print "Summary of Statistics"
        print "---------------------"
        built_in_lst = ['Number of Observations','Mean','Variance','Standard Deviation',
                        'Minimum', 'Index of Minimum', 'Maximum', 'Index of Maximum']        
        sum_df = self.summary_df[built_in_lst+self.user_defined_val_name_lst]
        num_cols = len(sum_df.columns)
        pd.set_option('display.max_columns',ncols)
        pd.set_option('display.width',width)
        # Print Summary in text (Built-in)
        def print_table(built_in_lst):
            for index in range(0,len(built_in_lst)+1,ncols):
                if (index != 0):
                    dis_header = built_in_lst[index-ncols:index]
                    #print "header",dis_header
                    sum_df_dis = sum_df[dis_header].copy()
                    sum_df_dis.rename(columns={'Number of Observations': '# Obs', 
                                               'Standard Deviation': 'Standard Dev.',
                                               'Index of Minimum':'Index(Min)',
                                               'Index of Maximum':'Index(Max)'}, 
                                      inplace=True)
                    print sum_df_dis.round(decimal_places)
                    print ""
            if (float(len(built_in_lst)) % float(ncols) != 0):
                # print last set of built-ins
                dis_header = built_in_lst[-int(float(len(built_in_lst)) % float(ncols)):]
                #print "header",dis_header
                sum_df_dis = sum_df[dis_header]
                sum_df_dis.rename(columns={'Number of Observations': '# Obs', 
                                            'Standard Deviation': 'Standard Dev.',
                                            'Index of Minimum':'Index(Min)',
                                            'Index of Maximum':'Index(Max)'}, 
                                  inplace=True)
                print sum_df_dis.round(decimal_places)
            print ""
            return
        print_table(built_in_lst)
        # Print user-defined stats
        if (len(self.user_defined_val_name_lst) != 0):
            print ""
            print "-----------------------"
            print "User-defined Statistics"
            print "-----------------------"
            print_table(self.user_defined_val_name_lst)
        pd.set_option('display.width', 80)
        pd.set_option('display.max_columns',20)
        return #self.summary_df[built_in_lst+self.user_defined_val_name_lst]
           
    def print_summary_to_latex(self, path, max_ncols = 4, max_nrows= 5, decimal_places=2, sparkline=True):
        #print self.summary_df
        if (sparkline):
            def sparkline_gen(x, file_name, save = False):
                import matplotlib.pyplot as plt
                import numpy as np
                #% matplotlib inline
                x = np.array(x)
                file_name = file_name.replace(' ','_')
                fig, ax = plt.subplots(1,1,figsize=(10,3))
                plt.plot(x, color='k')
                plt.plot(len(x)-1, x[-1], color='r', marker='o')
                # remove all the axes
                for k,v in ax.spines.items():
                    v.set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                if (save):
                    plt.savefig(file_name,bbox_inches='tight',transparent=True)
                else: 
                     plt.title(file_name)
                     plt.show()
                return fig
        import matplotlib.pyplot as plt
        #print self.summary_df.round(decimal_places).T
        sum_df = self.summary_df.transpose().round(decimal_places)
        latex_head = '\\def\\arraystretch{1.5}\n'
        table_head = '\\begin{table}[htbp]\n' 
        table_head += '\\centering\n\n'
        
        latex_tail  = '\\hline\n'
        latex_tail += '\\end{tabular}\n\\end{table}'
        
        # generate tabular columns lst and row lst
        ncols = len(self.summary_df)
        end = ncols
        if (max_ncols > ncols):
            max_ncols = ncols
        
        nrows = len(self.summary_df.columns)
        col_start_lst = range(0,ncols,max_ncols)
        col_end_lst = range(max_ncols,ncols,max_ncols)+[ncols]
        col_lst = list(sum_df.columns)
        
        # generate row lst
        nrows = len(sum_df)
        if (max_nrows > nrows):
            max_nrows = nrows
        #print "nrows:",nrows
        row_start_lst = range(0,nrows,max_nrows)
        row_end_lst = range(max_nrows,nrows,max_nrows)+[nrows]
        
        code_lst = []
        #print "Sum_df"
        #print sum_df
        int_lst = ['Number of Observations', 'Index of Minimum', 'Index of Maximum']
        for col_start, col_end in zip(col_start_lst, col_end_lst):
            #print "start:",col_start
            #print "end  :",col_end
            dis_df_col = sum_df[col_lst[col_start:col_end]]
            header_tabular = '\\begin{tabular}{l'+'l'*len(dis_df_col.columns)+'}\n'
            header_content = "\\hline\n"+"&"
            for index in dis_df_col.columns:
                header_content += "\\bf "+index+"&"
            header_content  = header_content[:-1]+"\\\\\n"
            header_content += "\\hline\n"
            content_lst = []
            for row_start, row_end in zip(row_start_lst, row_end_lst):
                content = ""
                dis_df = dis_df_col.iloc[row_start:row_end]
                row_index_lst = list(dis_df.index)
                for row_index in row_index_lst:
                    content += row_index+"&"
                    for col in dis_df.columns:
                        if (row_index in int_lst):
                            content += str(int(dis_df.loc[row_index].loc[col]))+"&"
                        else:
                            content += str(dis_df.loc[row_index].loc[col])+"&"
                    content = content[:-1]
                    content += "\\\\\n"
                #content = latex_head + header_tabular + header_content + content
                #print content
                content_lst.append(content)
            code_lst.append((header_tabular,header_content,content_lst,len(dis_df_col)))

        # Generate Sparkline
        if (sparkline):
            sparkline_lst = []
            for col in self.df.columns:
                sparkline_lst.append((col, col+'_sparkline'))
        # Update Tex Code
        tex_content = latex_head
        counter = 0
        for code in code_lst:
            header_tabular = code[0]
            header = code[1]
            content_lst = code[2]
            for content in content_lst[:-1]:
                tex_content += table_head + header_tabular + header+content+latex_tail+'\n\n'
            tex_content += table_head + header_tabular + header+content
            if (sparkline):
                tex_content += "\\hline\n"
                tex_content += 'Sparkline&'
                for index in range(col_start_lst[counter], col_end_lst[counter]):
                    img_cmd = "\\includegraphics[scale=0.1]{"
                    tmp = sparkline_lst[index][1]
                    img_path = "\""+tmp.replace(' ','_')+"\""
                    tex_content += img_cmd+img_path+"}"+"&"
                tex_content = tex_content[:-1]+"\\\\\n"
            tex_content += latex_tail+"\n\n"
            counter += 1
            
        # Try create a directory and file for output
        # if not successful, return a dirtionary of code and graphs 
        import os
        path += "summary/"
        try: 
            os.makedirs(path)
        except OSError:
            if (not os.path.isdir(path)):
                import warnings
                msg  = "Cannot create "+path+" and the directory does not exist."
                msg += "Returned a dictionary d where d['tex'] contains the tex code and "
                msg += "d['sparkline'] contains a list of tuples (column name, file name, matplotlib axis"
                msg += " of sparklines)."
                warnings.warn(msg)
                rt = dict()
                rt['tex'] = tex_content
                rt['sparkline'] = sparkline_lst
                for sparkline_tuple in sparkline_lst:
                    col = sparkline_tuple[0]
                    sparkline_gen(self.df[col],path+sparkline_tuple[1], save= False)
                return rt
        # Output tex_content
        with open(path+'tex.tex', 'w') as outfile:
            outfile.write(tex_content)
        rt = dict()
        rt['tex'] = tex_content
        if (not sparkline):
            return rt
        # Output sparklines
        for sparkline_tuple in sparkline_lst:
            col = sparkline_tuple[0]
            sparkline_gen(self.df[col],path+sparkline_tuple[1], save= True)
        rt['sparkline'] = sparkline_lst
        return tex_content

    def output_pdf_summary(self, path, file_name="summary", max_ncols = 4, max_nrows= 5, sparkline=True):
        tex_content = self.print_summary_to_latex(path, max_ncols = 4, max_nrows= 5, sparkline=True)
        path += "summary\\"
        latex_file_head = '''
\\documentclass[12pt]{article}
\\usepackage[english]{babel}
\\usepackage[utf8]{inputenc}
\\usepackage[scaled=0.85]{beramono}
\\usepackage[margin=0.8in]{geometry}
\usepackage{graphicx}
\\begin{document}'''
        latex_file_tail = '\\end{document}'
        tex_content = latex_file_head + tex_content + latex_file_tail
        #print tex_content
        #print file_name
        f = open(path+file_name+".tex", "w")
        f.write(tex_content)
        f.close()       
        import os  
        cwd = os.getcwd()+"\\"+"summary\\"
        cmd = "pdflatex" + " "+ "\""+ cwd + file_name+".tex" + "\""       
        #print cmd        
        return os.system(cmd)
