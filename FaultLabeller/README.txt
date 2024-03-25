Project Title: Data Annotation Tool for Big Data (with Machine Learning)
Developer: Sean Doherty
Supervisor: Robert Atkinson
Academy body: University of Strathclyde
Student Number: 202013008

Motivation: ML has incredible potential to implement fault prediction systems in the Industiral IoT.  However, models requires vast amounts of well labelled data.
    Fault labelling is a laborious, repetitive and intensive task, which introduces great risk of human error. 
    This software aims to optimises the fault labelling process by cutting time and improving accuracy.

What it does: The software include multiple visualtions of data to enable plant engineers to label faults, 
    and uses ML to detect faults, cutting the time and effort required of engineers.

Technologies used: the UI is implemented in Pythono Dash.
    Data handling is provided by Pandas
    ML algorithms (PCA, K-Means, Autoencoding, DBSCAN and Multiclass NN use SKLearn)

How to Run:
    - Open project in a suitable IDE. VSCode is recommended
    - Open the folder 'Fault Labeller'
    - Install neceasry libraries. This will include
        - Keras
        - Python Dash
        - Pandas
    - Run 'Main.py'
    - Open terminal. If a library is missing, terminal will indicate what must be installed.
    - Alongside other text, the terminal will display the line 'Dash is running on http://<IP ADDRESS>'
    - Copy and paste / Control click to open the program in browser


File Stucture:
    All project files/folders are contained with the folder 'Fault Labeller'
    'Main.py' starts the application and includes all callback functions
    Internal Libraries includes 
        - 'Components.py' which has all Python Dash Components
        - 'Layout.py' which arranges those Components
        - 'Styles' which holds componetn styling
        - 'ML_Functions.py' with all ML Functions (shocker)
    Neural Networks 
        - Saves the most recent autoencoder
        - Saves the most recent multiclass neural Network
    Data 
        - Had four CSVs of unlabelled data and four CSVs of labelled data
        takan from the Tenesse Easmen chemical process [https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset]
        
        SCENARIO 1
        - Normal operation for 480 samples
        - Fault 1 for 480 samples
        - Normal operation 480 samples
        - Fault 2 for 480 samplels
        - Normal operation for 480 samples
        - Fault 3 for 480 samples
        
        Scenario 2
        - Normal operation 100 samples
        - Fault 1 for 20 samples
        - Normal operation 100 samples
        - Fault 2 for 20 samples
        - Normal operation 100 samples
        - Fault 3 for 20 samples
        - Repeated three times

        Scenario 3
        - Normal operation for 480 samples
        - Fault 3 for 300 samples
        - Fault 1 for 300 samplels
        - Fault 2 for 300 samples
        - Repeated twice times

        Scenario 4
        - Normal operation for 1440 samples
        - Fault 2 for 256 samples
        - Normal operation for 2160 samplels
        - Fault 1  for 256 samples
        - Fault 3 for 256 samples

Happy labelling!