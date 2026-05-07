# Modelling Multi-Class Component Availability for Dependable Cyber-Physical System Adaptation

## Abstract
Cyber-Physical Systems (CPS) operate in environments where constituent components (e.g., vehicles in EV charging infrastructures, human caretakers in assistive care scenarios) are not always available. Planning the best way of performing tasks in such systems in a dependable manner is not always straightforward due to uncertainty in such temporal availability constraints. Existing work has addressed this issue employing combinations of diverse techniques, such as bio-inspired algorithms and quantitative verification. However, these approaches characterize the uncertainty in the availability of system components in a homogeneous way (e.g., they model variability in periods of availability across different classes of components with the same probability distributions). 
In this paper, we present what is, to the best of our knowledge, the first planning approach for CPS able to reduce uncertainty in temporal availability constraints to improve the quality of plans. This is achieved by considering heterogeneous uncertainty profiles for different classes of system components and using clustering techniques to automatically allocate components to classes. Our approach is evaluated on two case studies from different domains. Results show that the proposed approach consistently and significantly outperforms---with negligible computational overhead---a baseline planner that employs uniform uncertainty assumptions, even when subjected to noisy data.

## Dependencies

The project is implemented using the latest version of Python (3.12.1) and Java (JDK23). 
The Java project relies on the following libraries and tools to function correctly:
1. **Jython**: A Java implementation of Python, required to bridge the execution of Python scripts within the Java environment. Ensure that `jython-standalone-X.X.X.jar` is included in the project build path.
2. **PRISM Library**: The project utilizes PRISM, a probabilistic model checker, including its components such as `Prism`, `PrismLog`, and related modules for parsing models (`parser.ast`), simulating modules (`simulator`), and handling properties files. To execute the project you must have the PRISM libraries integrated and accessible in the project setup. For more information check https://github.com/prismmodelchecker/prism-api and https://github.com/prismmodelchecker/prism prism.

Python files require the installation of the following libraries for its development and execution:
1. **NumPy** 1.26.2
2. **Matplotlib** 3.8.2

## Repository structure
This repository contains the following items:
* `Readme.md`: this file explaning the code of the project
* `Python_Files`: this folder contains six files where we can find the clases and the functions needed to execute the algorithm.  
  * `Genetic_algorithm.py `: this file contains the code of the original genetic algorithm, that solve the vehicle charging planning problem considering uncertainty.
  * `Classes_generator.py`: this file contains the functions responsible for generating the specific resource and consumer classes required by the algorithms.
  * `Generated_classes.py`: this file includes an example of the generated classes for the vehicles scenario.
  * `ClassesAG.py`: this file contains the functions that the new Genetic Statistical Model Checking Algorithm (GSMCA) needs to work.
  * `RunGA.py`: this file contains the code to run the new version of the genetic algorithm using Python.
  * `Clustering.py`: this file contains the code to run the clustering algorithms.
* `JavaAlgorithms`: this folder contains the Java project. In the src package we can find two files:
  * `ExecuteGeneticAlgorithm.java`: this file handles the execution of the GSMCA.
  * `ModelCheckFromFiles.java`: this file is responsible for evaluate the model launching PRISM. It consist on a version of a PRISM API example adapted to our project.
* `Clustering_Experiments:` this folder contains the Clustering_experiments.ipynb file, which includes the code to apply and compare clustering methods (KMeans and DBSCAN)s.
* `Data_Generation:` this folder contains the Data_generator.ipynb file, which id needed to create synthetic datasets, with configurable separation levels between groups, noise, and punctuality profiles.
* `Experiments_Visualization:` this folder contains the Data_generator.ipynb file, which id needed to create synthetic datasets, with configurable separation levels between groups, noise, and punctuality profiles.
* `Experiments_Visualization:` in this folder we can find the Visualization.ipynb file, which contains the code to simulate some experiments and generate the graphs.
* `Data:` This folder contains the data and configuration files used to run the experiments. It includes two subfolders:
  * `Original_Data`: Contains the CSV data files used as input for the clustering algorithms.
  * `Clustering_Results`: Contains CSV files with clustered data generated for different scenarios. These files are the results of the clustering process using different methods (DBSCAN and K-means). They are used as input for the genetic algorithm.

## Configuration File Structure
The configuration files define a standardized schema for modeling resource allocation scenarios, such as electric vehicle charging or robotic patient feeding. Each file describes two main classes: a **Consumer** (e.g., ElectricVehicle, Patient, Crop) and a **Resource** (e.g., Charger, Robot), including their attributes and methods.

The configuration sets key parameters like the number of entities and resources, their states (AVAILABLE, ACTIVE, DONE), progress tracking variables, speed and capacity values, and action labels (start_action, release_action, action) that drive transitions in the PRISM model. It also includes reward options (e.g., reward_acum, reward_timespan) and specifies the output file name.

This modular design allows the generator to easily adapt to different domains by simply changing the input file, supporting flexibility, reusability, and scalability.

## Running the Experiments
To run the code, you need to do it within Eclipse IDE for Java and Visual Studio Code for Python or similar environment.The following explains how to run algorithms, with the environments previously installed:

### Download and import the Java project from GitHub into Eclipse
1. First, go to the GitHub repository, click on Code, and select Download ZIP. Extract the downloaded ZIP file to your preferred folder. Alternatively, if you have Git installed, you can clone the repository by using the command "git clone" <repository URL> in your terminal.

2. Once you have the project files, open Eclipse and navigate to File > Import > Existing Projects into Workspace. In the dialog that appears, choose Select root directory and browse to the folder where you extracted the project. Then, click Finish to import it into Eclipse.

3. If the project is not recognized as a Java Project, right-click on it in the Project Explorer, go to Configure > Convert to Java Project, and Eclipse will set it up as a Java Project. Additionally, make sure all required dependencies are configured in the Build Path to avoid errors.

### Synthetic Data Generation
The synthetic datasets used in this project simulate realistic scenarios with configurable consumer behaviors, schedules, conditions, and usage patterns. Different dataset variants are generated to evaluate clustering robustness, including datasets with varying group separation levels and configurable noise. These datasets are later used as input for the clustering algorithms and the genetic optimization process.

### Execution of the clustering algorithms
You can execute the clustering experiments using the code found in Clustering_experiments.ipynb. To do so, open the notebook in Google Colab, Jupyter Notebook, or a similar environment and run all cells in order. The notebook will load the input data from the Vehicles_Data/ folder, apply clustering methods (such as K-Means) , and display the corresponding visualizations and comparisons. If you want to modify the clustering parameters (for example, the number of clusters, distance metric, or initialization method) or use different input data, you can edit the configuration variables defined at the beginning of the code. Once executed, you can export the clustered results using "df.to_csv('Vehicles_Data/vehicles_clustered.csv', index=False)", and the generated clusters can later be used to execute the main algorithm and visualization scripts located under Experiments_Visualization/ (as explained in the following sections)..

### Running the Genetic Algorithm
1. **Select the file ExecuteGeneticAlgorithm.java**
   - Ensure the correct file is open.
     
2. **Verify the Paths to the Python Files**
   - Double-check that the lines executing the Python scripts and functions use the correct paths in the following format:
     ```java
     interpreter.execfile("C:\\Users\\usuario\\Escritorio\\project\\geneticAlgorithm.py");
     ```
   - Ensure the Python files (ClassesAG.py and runGA.py) exist in the specified location.
     
3. **Compile and Run the Program**
   - Select **Run As > Java Application**.

### Execution of the code to generate the charts
You can also execute some of the experiments and generate charts using the code found in Visualization.ipynb. To achieve this, simply run the cells in Google Colab or similar environment in order, and various graphs will be displayed as output. If you wish to modify the data or the parameters to create new experiments and graphs, you can do so by editing the parameters and the data structures at the beginning of the code that generates each graph.
