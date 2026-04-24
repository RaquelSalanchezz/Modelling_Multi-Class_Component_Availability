//==============================================================================
//	
//	Copyright (c) 2017-
//	Authors:
//	* Dave Parker <d.a.parker@cs.bham.ac.uk> (University of Birmingham)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================

//==============================================================================
//	
//	Copyright (c) 2017-
//	Authors:
//	* Dave Parker <d.a.parker@cs.bham.ac.uk> (University of Birmingham)
//	
//------------------------------------------------------------------------------
//	
//	This file is a version of a part of PRISM API.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================

//==============================================================================
//	
//	Copyright (c) 2017-
//	Authors:
//	* Dave Parker <d.a.parker@cs.bham.ac.uk> (University of Birmingham)
//	
//------------------------------------------------------------------------------
//	
//	This file is a version of a part of PRISM API.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================

package demos;


import prism.Prism;
import prism.PrismDevNullLog;
import prism.PrismException;
import prism.PrismFileLog;
import prism.PrismLog;
import prism.Result;
import prism.UndefinedConstants;



import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collections;

import javax.swing.event.ListSelectionEvent;

import parser.ast.ModulesFile;
import parser.ast.PropertiesFile;
import simulator.ModulesFileModelGenerator;
import simulator.SimulatorEngine;
import simulator.method.CIwidth;
import simulator.method.SimulationMethod;

import java.util.List;
import java.util.ArrayList;




/**
 * Class to control PRISM programmatically,
 * through the functions exposed by the class prism.Prism.
 * 
 * This shows how to load a model from a file and model check some properties,
 * either from a file or specified as a string, and possibly involving constants. 
 * 
*/


public class ModelCheckFromFiles {	
	public static void main(String[] args)
	{
		new ModelCheckFromFiles().run();
	}

	// List to save the results
	public List<Double> run()
	{
		double cost = -1;
		double timespan = -1;
		double ncompleted = -1;
		List<Double> propiedades = new ArrayList<>();
		
		try {
			
			PrismLog mainLog = new PrismFileLog("stdout");
			Prism prism = new Prism(mainLog);
			prism.initialise();
			
			//Set the model path
			ModulesFile modulesFile = prism.parseModelFile(new File("C:\\Users\\raque\\OneDrive\\Escritorio\\prueba/general_opt_system\\models\\modelomodelohey.prism"));
			
			//Set the properties file path
			PropertiesFile propertiesFile = prism.parsePropertiesFile(modulesFile, new File("C:\\Users\\raque\\OneDrive\\Escritorio\\propiedades.pctl"));
			
			SimulatorEngine simEngine = new SimulatorEngine(prism);
			ModulesFileModelGenerator modelGen = new ModulesFileModelGenerator(modulesFile, prism);
			simEngine.loadModel(modelGen, modelGen);
			
			SimulationMethod simMethod = new CIwidth(0.05, 10);
			
			//To obtain only the first property. Uncomment if it is neccesary
			//simEngine.modelCheckMultipleProperties(propertiesFile, Collections.singletonList(propertiesFile.getProperty(0)), null, 10000000, simMethod); 
			
			
			
			// Evaluate properties
            Object result = simEngine.modelCheckMultipleProperties(
                propertiesFile, 
                Collections.singletonList(propertiesFile.getProperty(0)), 
                null, 
                10000000, 
                simMethod
            );

            // Process the result
            if (result instanceof Object[]) {
                Object[] resultsArray = (Object[]) result;

               
                if (resultsArray.length > 0 && resultsArray[0] instanceof prism.Result) {
                    prism.Result propertyResult = (prism.Result) resultsArray[0];

                    // Extract the numeric value of the result
                    Object rawValue = propertyResult.getResult(); 
                    if (rawValue instanceof Number) {
                        cost = ((Number) rawValue).doubleValue();
                        System.out.println("Costo evaluado: " + cost);
                    } else {
                        System.err.println("El valor del resultado no es numérico: " + rawValue);
                    }
                } else {
                    System.err.println("El resultado no contiene un objeto Result válido.");
                }
            } else {
                System.err.println("El resultado devuelto no es un array.");
            }
            
            //--------------------------------------------------
            
            // Evaluate properties
            Object result2 = simEngine.modelCheckMultipleProperties(
                propertiesFile, 
                Collections.singletonList(propertiesFile.getProperty(1)), 
                null, 
                10000000, 
                simMethod
            );

            // Process the result
            if (result2 instanceof Object[]) {
                Object[] resultsArray = (Object[]) result2;

                // Verify that the list is not empty and handle the first element
                if (resultsArray.length > 0 && resultsArray[0] instanceof prism.Result) {
                    prism.Result propertyResult = (prism.Result) resultsArray[0];

                    // Extract the numeric value of the result
                    Object rawValue = propertyResult.getResult(); // Este es el valor del resultado
                    if (rawValue instanceof Number) {
                        timespan = ((Number) rawValue).doubleValue();
                        System.out.println("Timespan evaluado: " + timespan);
                    } else {
                        System.err.println("El valor del resultado no es numérico: " + rawValue);
                    }
                } else {
                    System.err.println("El resultado no contiene un objeto Result válido.");
                }
            } else {
                System.err.println("El resultado devuelto no es un array.");
            }
            
            
            
            //--------------------------------------------------
            
            // Evaluate properties
            Object result3 = simEngine.modelCheckMultipleProperties(
                propertiesFile, 
                Collections.singletonList(propertiesFile.getProperty(2)), 
                null, 
                10000000, 
                simMethod
            );

            // Process the result
            if (result3 instanceof Object[]) {
                Object[] resultsArray = (Object[]) result3;

                // Verify that the list is not empty and handle the first element
                if (resultsArray.length > 0 && resultsArray[0] instanceof prism.Result) {
                    prism.Result propertyResult = (prism.Result) resultsArray[0];

                    // Extract the numeric value of the result
                    Object rawValue = propertyResult.getResult(); // Este es el valor del resultado
                    if (rawValue instanceof Number) {
                        ncompleted = ((Number) rawValue).doubleValue();
                        // FIX Bug 8: Was printing timespan variable instead of ncompleted
                        System.out.println("Num completed evaluado: " + ncompleted);
                    } else {
                        System.err.println("El valor del resultado no es numérico: " + ncompleted);
                    }
                } else {
                    System.err.println("El resultado no contiene un objeto Result válido.");
                }
            } else {
                System.err.println("El resultado devuelto no es un array.");
            }
            
            //--------------------------------------------------
            
            
			simEngine.createNewOnTheFlyPath();
			simEngine.initialisePath(null);
			simEngine.automaticTransitions(100, false);

			
		} catch (PrismException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		
		propiedades.add(cost);
		propiedades.add(timespan);
		propiedades.add(ncompleted);
		
		//Return the results
		return propiedades;
	}	
	
}
