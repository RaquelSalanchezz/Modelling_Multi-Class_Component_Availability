package demos;
import org.python.util.PythonInterpreter;
import org.python.core.PyObject;

import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.python.core.PyList;
import org.python.core.PyTuple;
import org.python.core.PyInteger;
import org.python.core.PyFloat;

import java.util.Random;


public class ExecuteGeneticAlgorithm {

    /**
     * Runs the clustering pipeline as an external Python 3 process.
     * Equivalent to how PRISM is called externally.
     *
     * @param python3Path      Full path to the Python 3 executable (e.g. "python" or "C:\\Python39\\python.exe")
     * @param scriptPath       Full path to clustering.py
     * @param outputKmeansCsv  Full path where the KMeans clustered CSV will be saved
     * @param outputDbscanCsv  Full path where the DBSCAN clustered CSV will be saved
     * @param scenario         Scenario to run: "vehicles" or "patients"
     */
    public static void runClustering(String python3Path, String scriptPath,
                                     String outputKmeansCsv, String outputDbscanCsv,
                                     String scenario) {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                python3Path, scriptPath,
                outputKmeansCsv, outputDbscanCsv,
                scenario
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // Print clustering output to console
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("[Clustering] " + line);
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                System.err.println("ERROR: Clustering script exited with code " + exitCode);
            } else {
                System.out.println("Clustering completed successfully. Output: " + outputDbscanCsv);
            }

        } catch (Exception e) {
            System.err.println("ERROR launching clustering process: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        // =====================================================================
        // PATHS — modify as needed
        // =====================================================================
        String basePath      = "C:\\Users\\raque\\OneDrive\\Escritorio\\prueba\\general_opt_system\\";
        String python3Path   = "python";   // or full path e.g. "C:\\Python39\\python.exe"
        String clusterScript      = basePath + "clustering.py";
        String clusteredKmeansCsv = basePath + "vehicles_case\\vehicles_data\\vehicles_clustered_kmeans.csv";
        String clusteredDbscanCsv = basePath + "vehicles_case\\vehicles_data\\vehicles_clustered_dbscan.csv";
        String scenario           = "vehicles";  // "vehicles" or "patients"

        // =====================================================================
        // STEP 0: RUN CLUSTERING — produces vehicles_clustered_with_std.csv
        // Must run before the GA so runGA.py reads the file with assigned_std
        // =====================================================================
        System.out.println("=== Running clustering (kmeans + dbscan) ===");
        runClustering(python3Path, clusterScript, clusteredKmeansCsv, clusteredDbscanCsv, scenario);

        // =====================================================================
        // Initialise Python interpreter (Jython) for the GA
        // =====================================================================
        PythonInterpreter interpreter = new PythonInterpreter();

        // Add directory to sys.path
        interpreter.exec("import sys");
        interpreter.exec("sys.path.append('" + basePath.replace("\\", "\\\\") + "')");
        
        
        // ------------Read configuration file and generate essential classes -------------------
        //---------------------------------------------------------------------------------------
        interpreter.execfile(basePath + "classes_generator.py");
        
        
        // -------------------------------- GENERATE INITIAL POPULATION ------------------------
        //---------------------------------------------------------------------------------------
        interpreter.execfile(basePath + "runGA.py");
        
        // Obtain the variable "population"
        PyObject poblacionPy = interpreter.get("population");
        PyObject cargadoresPy = interpreter.get("resources");

        // Print initial population
        System.out.println("POBLACION INICIAL");
        String modelo = "modelo";
        
        // Iterate the population
        for (PyObject solucionPy : poblacionPy.asIterable()) {
        	for (PyObject itemPy : solucionPy.asIterable()) {
                PyObject vehiculo = itemPy.__getattr__("consumer");
                PyObject cargador = itemPy.__getattr__("resource");

                int idVehiculo = vehiculo.__getattr__("id").asInt();
                int idCargador = cargador.__getattr__("id").asInt();
                double tiempoInicio = itemPy.__getattr__("begin_time").asDouble();
                double tiempoFin = itemPy.__getattr__("end_time").asDouble();

                System.out.println("Identificador Vehículo: " + idVehiculo);
                System.out.println("Inicio: " + tiempoInicio);
                System.out.println("Fin: " + tiempoFin);
                System.out.println("Cargador: " + idCargador);
                System.out.println(".....");
            }
        }
        
        
        //------------------------- GENETIC ALGORITHM --------------------------
        //----------------------------------------------------------------------
        
        PyObject clasesAG = interpreter.get("clasesAG");
        PyObject generatePrismModel = clasesAG.__getattr__("generate_prism_model"); 
        PyObject generateEvaluationPrismModel = clasesAG.__getattr__("generate_evaluation_prism_model"); 
        PyObject generateConfEvalModel = clasesAG.__getattr__("generate_evaluation_model_config"); 
        
        for (int num = 0; num < 5; num++) {
        	
        	//-----------------------EVALUATE POPULATION---------------------------------
        	
        	PyList evaluacionesPython = new PyList();
        	
        	for (PyObject solucionPython : poblacionPy.asIterable()) {
        		
                PyObject resultadoModelo = generatePrismModel.__call__(solucionPython);
                
                List<Double> propiedades = new ModelCheckFromFiles().run();
                
                double cost = propiedades.get(0);
                System.err.println("El resultado devuelto coste es " + cost);
                
                double timespan = propiedades.get(1);
                System.err.println("El resultado devuelto coste es " + timespan);
                
                if (cost >= 0 && timespan >= 0) {
	                evaluacionesPython.add(new PyTuple(new PyObject[]{
	                    solucionPython, new PyFloat(cost), new PyInteger(0), new PyFloat(timespan)
	                }));  
                }
        	} 
        	
        	PyList nuevaPoblacion = new PyList();
        	PyList listaPoblacion = (PyList) poblacionPy;
        	
        	for (int j = 0; j < listaPoblacion.size(); j++) {
	        	
        		//------------------------- PARENTS SELECTION ----------------------------
	        	
	            PyObject seleccionarPadres = clasesAG.__getattr__("parents_selection");
	            PyObject padres = seleccionarPadres.__call__(evaluacionesPython);
	            
	            PyList padresList = (PyList) padres;
	
	            System.out.println("Padres seleccionados:");
	            for (Object padre : padresList) {
	                System.out.println(padre);
	            }
	            
	            //------------------------ PARENTS CROSSOVER -----------------------------
	            
	            Random random = new Random();
	            PyObject padre1 = (PyObject) padresList.get(random.nextInt(padresList.size()));
	            PyObject padre2 = (PyObject) padresList.get(random.nextInt(padresList.size()));
	            
	            PyObject cruzarPadres = clasesAG.__getattr__("parents_crossover");
	            PyObject hijo = cruzarPadres.__call__(padre1, padre2, cargadoresPy);
	            
	            //----------------------- MUTATION INTRODUCTION --------------------------
	            PyObject mutarHijo = clasesAG.__getattr__("mutate_son");
	            PyObject hijoMutado = mutarHijo.__call__(hijo, new PyFloat(0.2), cargadoresPy);
	            
	            System.out.println("Hijo mutado:");
	            System.out.println(hijoMutado);
	            
	            nuevaPoblacion.add(hijoMutado);
        	}
        }
        
        
        //------------------------------ SHOW BETTER SOLUTION --------------------------------
        //-----------------------------------------------------------------------------------
        
        PyList evaluacionesPython = new PyList();
    	
    	for (PyObject solucionPython : poblacionPy.asIterable()) {
    		
            PyObject resultadoModelo = generatePrismModel.__call__(solucionPython);
            
            List<Double> propiedades = new ModelCheckFromFiles().run();
            
            double cost = propiedades.get(0);
            System.err.println("The result for cost is " + cost);
            
            double timespan = propiedades.get(1);
            System.err.println("The result for timespan is " + timespan);
            
            if (cost >= 0 && timespan >= 0) {
	            evaluacionesPython.add(new PyTuple(new PyObject[]{
	                solucionPython, new PyFloat(cost), new PyInteger(0), new PyFloat(timespan)
	            }));  
            }
    	} 
    	
	    //-------------------- BETTER SOLUTION SELECTION ----------------------------
	    
	    PyObject seleccionarPadres = clasesAG.__getattr__("parents_selection");
	    PyObject padres = seleccionarPadres.__call__(evaluacionesPython);
	            
	    PyList padresList = (PyList) padres;
	
	    System.out.println("Parents:");
	    for (Object padre : padresList) {
	         System.out.println(padre);
	    }
	    
	    PyObject mejorSol = (PyObject) padresList.get(0);
	    
	    System.out.println("------------BETTER SOLUTION--------------");
	    System.out.println("----------------------------------------");
	    for (PyObject itemPy : mejorSol.asIterable()) {
            PyObject vehiculo = itemPy.__getattr__("consumer");
            PyObject cargador = itemPy.__getattr__("resource");

            int idVehiculo = vehiculo.__getattr__("id").asInt();
            int idCargador = cargador.__getattr__("id").asInt();
            double tiempoInicio = itemPy.__getattr__("begin_time").asDouble();
            double tiempoFin = itemPy.__getattr__("end_time").asDouble();

            System.out.println("Id: " + idVehiculo);
            System.out.println("Begin: " + tiempoInicio);
            System.out.println("End: " + tiempoFin);
            System.out.println("Resource: " + idCargador);
            System.out.println(".....");
        }
	    
	    PyObject resultadoModelo = generateConfEvalModel.__call__(mejorSol);
	    
        List<Double> propiedades = new ModelCheckFromFiles().run();
        
        double cost = propiedades.get(0);
        System.err.println("Cost/Disruption of better solution: " + cost);
        
        double timespan = propiedades.get(1);
        System.err.println("Timespan of better solution: " + timespan);
        
        double ncompleted = propiedades.get(2);
        System.err.println("Number of unsatisfactory tasks: " + ncompleted);
              
        interpreter.close();
    }
}
