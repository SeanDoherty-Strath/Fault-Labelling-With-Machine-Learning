DATA ANNOTATION TOOL FOR BIG DATA

Developer: Sean Doherty

Supervisor: Dr. Robert Atkinson

Academic body: University of Strathclyde

Student Number: 202013008

One of the greatest barriers to the Industrial Internet of Things is harvesting accurate training data: especially of faults. 

Fault labelling is a time consuming and repetitive process, which ingrains blindspots & biases in the data. 

There is huge demand for a software which can

	- minimise the time & reffort required from a plant's Control Room to report faults
 
 	- maximise the accuracy of training data in the IoT

This software achieves this in three ways...

1) Intuitive fault labelling

![image](https://github.com/SeanDoherty-Strath/Scripts/assets/113599995/b6f0be35-54cb-4cd4-ad42-ec33a808ae8c)

2) Auto detection of faults, powered by unsupervised machine learning
![image](https://github.com/SeanDoherty-Strath/Scripts/assets/113599995/95f1c731-9349-4b72-9c19-d3c59926601f)

3) Progressively smarter detection, powered by semi-supervised AI, and intelligent visualition of data
![image](https://github.com/SeanDoherty-Strath/Scripts/assets/113599995/8d3f19bc-f50a-4062-be3b-e335c32b3545)



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
	
    - Copy and paste this address to open the program in your browser


File Stucture:

    All project files/folders are contained with the folder 'Fault Labeller'
	
    'Main.py' starts the application and includes all callback functions
    Internal Libraries includes 
	
        - 'Components.py' which has all Python Dash Components
		
        - 'Layout.py' which arranges those Components
		
        - 'Styles' which holds componetn styling
		
        - 'ML_Functions.py' with all ML Functions (suprise!)
		
    Neural Networks 
	
        - Saves the most recent autoencoder
		
        - Saves the most recent multiclass neural Network
		
    Data 
        - Had four CSVs of unlabelled data and four CSVs of labelled data taken from the Tenesse Easmen chemical process [https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset]
        for demo purposes
	
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
