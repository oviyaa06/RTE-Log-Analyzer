# RTE-Log-Analyzer

#Problem statement

•	RTE log files contain a massive volume of data, making analysis time-consuming.

•	Important parameter values are embedded within unstructured log messages.

•	Engineers must manually search through thousands of log entries to find relevant information.

•	Existing log viewers lack parameter-wise analysis and visualization capabilities.

•	Processing large log datasets can lead to performance and memory challenges during analysis.

#Proposed Solution

•	Developed a centralized RTE Log Analyzer to analyze large RTE datasets stored in Parquet files.

•	Allows users to select a time range and automatically load the relevant Parquet files.

•	Extracts and categorizes parameters from Core Input, Core Settings, and Core Output messages.

•	Converts extracted information into a structured Parquet dataset for efficient querying and analysis.

•	Provides interactive graphs and export capabilities while using memory-efficient streaming to handle large datasets.

#Files in Code

•	NGridAnalysisWindow.py acts as the main RTE Log Analyzer application, providing a graphical interface for loading Parquet files, selecting time ranges, extracting and categorizing parameters from RTE messages, searching parameters, generating interactive graphs, exporting data to CSV and Excel, and performing memory-efficient analysis of large datasets.

•	requirements.txt – Stores all Python package dependencies required to run the RTE Log Analyzer. 

#Functionalities

•	Time-Based Analysis – Analyze Parquet data within a user-selected start and end time range.

•	Automatic Parameter Extraction – Extracts parameters from Core Input, Core Settings, and Core Output categories.

•	Parameter Search & Selection – Supports category-wise browsing, search, and multi-parameter selection.

•	Interactive Graph Visualization – Displays parameter trends over time with zoom, pan, and hover support.

•	Multiple Graph Management – Allows creation and removal of up to four graph panels simultaneously.

•	Data Export & Progress Tracking – Exports selected data to CSV/Excel and provides real-time processing progress updates.
