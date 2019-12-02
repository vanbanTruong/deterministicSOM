/**
 * @func DeterministicSOM refers to a determinized self-organizing map (SOM), which provides an automatic 
 * data analysis technique to help produce a low-dimensional representation of the high-dimensional 
 * input space with random eliminators.
 */

package cloudType;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class DeterministicSOM {
	
	private static final double LOG_BASE= 10;
	
	private Dataset dataset; 
	private Node[][] map;
	private double initialLearningRate;
	private int maxNumIterations;
	private double mapRadius;
	private boolean randomSelection;
	private boolean convergable;
	
	public DeterministicSOM(Dataset dataset, 
			int rows, 
			int cols, 
			int numFeatures, 
			int maxNumIterations,
			double initialLearningRate,
			boolean randomSelection,
			boolean randomInit,
			boolean forceIterations){
		
		this.dataset= dataset;
		this.map= new Node[rows][cols];
		Node node;
		for (int i = 0; i < map.length; i++) {
			for (int j = 0; j < map.length; j++) {
				node= new Node(i, j, numFeatures);
				map[i][j]= node;
			}
		}
		
		if (randomInit) {
			randomInit();
		} else {
			gradientInit();
		}
		
		this.maxNumIterations= maxNumIterations;
		this.initialLearningRate= initialLearningRate;
		this.mapRadius= (double) Math.max(cols, rows)/2;
		this.randomSelection= randomSelection;
		this.convergable= !forceIterations;	
	}
	
	/**
	 * Map node prototype feature vectors are initialized from the top left
	 * corner to the bottom right corner from 0 to 1. Because of this
	 * static initialization scheme (that has been found to be better)
	 * rather than a random one
	 * the network will be the same under that same inputs and parameters
	 * so different settings can be compared more easily 
	 */
	
	private void gradientInit() {
		System.out.println("Using gradient initialization...");
		double maxDist = Math.pow(map.length-1, 2) + Math.pow(map.length>0?map[0].length-1:0, 2);
		for (int i = 0; i < map.length; i++) {
			for (int j = 0; j < map[i].length; j++) {
				map[i][j].initPrototype((Math.pow(i, 2) + Math.pow(j, 2)) / maxDist);
			}
		}
	}
	
	private void randomInit() {
		System.out.println("Using random initialization...");
		for (int i = 0; i < map.length; i++) {
			for (int j = 0; j < map[i].length; j++) {
				map[i][j].initPrototype(Math.random());
			}
		}
	}
	
	/**
	 * Gets the neighborhood radius squared for efficiency.
	 * @param iteration the current iteration
	 * @return the computed neighborhood radius squared
	 */
	public double getRadiusSq(int iteration, int max, double timeConstant) {
		return Math.pow(mapRadius* Math.pow(10, -iteration/timeConstant), 2);
	}
	
	/**
	 * Gets the learning rate given the iteration
	 * @param iteration the current iteration
	 * @return the computed learning rate
	 */
	public double getLearningRate(int interation, int max) {
		return initialLearningRate*Math.pow(10, -(double)interation/max);
	}
	
	/**
	 * Seeks out and returns the node that best matches the
	 * given feature vector.
	 * @param features a set of features to match against
	 * @return the best match unit/node
	 */
	private Node bestMatch(double []features) {
		Node bmu= null;
		double bmuDist= Double.MAX_VALUE;
		double newDist;
		for (int i = 0; i < map.length; i++) {
			for (int j = 0; j < map[i].length; j++) {
				newDist= map[i][j].distance(features);
				if (newDist< bmuDist) {
					bmu= map[i][j];
					bmuDist= newDist;
				}
			}
		}
		return bmu;
	}
	
	public double distanceSq(Node n1, Node n2) {
		return Math.pow((n2.row-n1.row), 2) + Math.pow(n2.col- n1.col, 2);
	}
	
	protected void train() {
		if (dataset.isEmpty()) {
			System.out.println("No data in dataset.");
			return;
		}
		
		// training type 1: random selection
		if (randomSelection) { 
			System.out.println("Using random selection...");
			if (!convergable) {
				System.out.println("Forcing all iterations...");
			}
			
			double radiusSq;
			double learningRate;
			double timeConstant= maxNumIterations/(Math.log(mapRadius)/Math.log(LOG_BASE));
			
			Dataset dsClone;
			boolean converged = false;
			boolean changed;
			int iter= 0;
			
		
			while (iter< maxNumIterations) {
				// setup training dataset
				dsClone= (Dataset) dataset.clone();
				
				// compute map values based on time
				radiusSq= getRadiusSq(iter, maxNumIterations, timeConstant);
				learningRate= getLearningRate(iter, maxNumIterations);
				
				if (convergable) {
					converged= true;
				}
				
				// show map every sample in random order
				while (!dsClone.isEmpty()) {
					changed= trainOnRecord(dsClone.remove((int)Math.round(Math.random()*(dsClone.size()-1))), 
							radiusSq, learningRate);
					if (convergable && changed) {
						converged= false;
					}
				}
				
				if (convergable && converged) {
					break; // converged so break main training loop
				}
				
				//print some status
				int iterNum= iter+1;
				if (iterNum % 10 == 0 || iterNum== 1) {
					System.out.println("Iteration: "+iterNum+" of "+maxNumIterations);
				}
				
				// increment iteration count
				iter++;
			}
			
			System.out.println((converged? "converged":"Finished")+"at iteration"+(converged? iter+1: iter)+"of"+ maxNumIterations+".");
			
		} else { // training type 2: staggered selection
			System.out.println("Using staggered selection...");
			if(!convergable) //if convergable initialized as true: not necessary to force all iterations
				System.out.println("Forcing all iterations...");
			
			double radiusSq;
			double learningRate;
			// note dataset.size() is just in place of maxNumIterations
			double timeConstant= dataset.size()/(Math.log(mapRadius)/Math.log(LOG_BASE));
			
			int frontIndex= 0;
			int backIndex= dataset.size()-1;
			
			boolean reverse= false;
			boolean converged= false;
			boolean changed;
			int iter= 0;
			
			while (frontIndex<= backIndex) {
				// compute map values based on time
				radiusSq= getRadiusSq(iter, dataset.size(), timeConstant);
				learningRate= getLearningRate(iter, dataset.size());
				
				int start= reverse? backIndex:frontIndex;
				int index= start;
				
				if (convergable) {
					converged= true;
				}
				
				do {
					changed= trainOnRecord(dataset.get(index), radiusSq, learningRate);
					
					if (convergable && changed) { 
						//if convergable is initialized as 'false', converged is always 'false', which means forcing all iterations
						converged= false;
					}
					
					index= (index+ (reverse?-1:1)+ dataset.size())% dataset.size();
					
				} while (start!= index);
				
				if (convergable && converged) {
					break; // converged so break main training loop
				}
				
				// move the index that was just used to start the iteration
				if (reverse) {
					backIndex--;
				}else {
					frontIndex++;
				}
				
				reverse= !reverse; // switch direction
				
				// print some status
				int iterNum = iter+1;
				if(iterNum % 10 == 0 || iterNum == 1) {
					System.out.println("Iteration: "+iterNum+" of "+dataset.size());
				}
				
				// increment iteration count
				iter++;
			}
			
			System.out.println(converged?"Converged":"Finished"+"at iteration"+ (converged?iter+ 1:iter)+" 0f "+ dataset.size()+".");
		}
	}
	
	private boolean trainOnRecord(Record record, double radiusSq, double learningRate) {
		Node bmu= bestMatch(record.getFeatures());
		
		int minRow= (int) Math.round(bmu.row- radiusSq);
		int minCol= (int) Math.round(bmu.col- radiusSq);
		int maxRow= (int) Math.round(bmu.row+ radiusSq);
		int maxCol= (int) Math.round(bmu.col+ radiusSq);
		
		if (minRow< 0) {
			minRow= 0;
		}
		if (minCol< 0) {
			minCol= 0;
		}
		if (maxRow> map.length- 1) {
			maxRow= map.length-1;
		}
		if (maxCol> map[0].length- 1) {
			maxCol= map[0].length- 1;
		}
		
		double distToNeighborSq;
		double influence;
		
		for (int i= minRow; i <= maxRow; i++) {
			for (int j = minCol; j <= maxCol; j++) {
				distToNeighborSq= distanceSq(bmu, map[i][j]);
				if (distToNeighborSq< radiusSq) {
					influence= Math.pow(LOG_BASE, -distToNeighborSq/(radiusSq*2));
					map[i][j].update(record.getFeatures(), learningRate, influence);
				}
			}
		}
		
		if (bmu!= record.bmu) { // found a new best match unit
			if (record.bmu!= null) {
				record.bmu.remove(record);
			}
			bmu.add(record);
			return true; //change bmu
		} else { //same old bmu
			return false; // no change
		}
		
	}
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		long startTime= System.currentTimeMillis();
		
		//Import data
		String dataFileName= "";
		CSVLoader loader= new CSVLoader();
		loader.setSource(new File(dataFileName));
		Instances data= loader.getDataSet();
		Instance instance= null;
		Dataset dataset= new Dataset();
		Record record;
		for (int i = 0; i < data.numInstances(); i++) {
			instance= data.instance(i);
			record= new Record(instance);
			dataset.add(record);
		}
		
		// Initialize experiment parameters
		int rows= 4;
		int cols=3;
		int numFeatures= instance.numAttributes();
		int maxNumIterations= dataset.size();
		double initialLearningRate= 0.5;
		boolean randomSelection= false;
		boolean randomInit= false; 
		boolean forceIterations= false;
		DeterministicSOM som= new DeterministicSOM(dataset, rows, cols, numFeatures, maxNumIterations, initialLearningRate, randomSelection, randomInit, forceIterations);
		
		som.train();
		
		String fileNameOfMapWeight= "MapWeight_"+ dataFileName;
		BufferedWriter bw= new BufferedWriter(new FileWriter("./data/results/"+ fileNameOfMapWeight));

		for (int i = 0; i < som.map.length; i++) {
			String content="";
			for (int j = 0; j < som.map[0].length; j++) {
				content= content.concat(som.map[i][j].getPrototype()[0]+",");
			}
			bw.write(content);
			bw.newLine();
		}
		
		bw.close();
		
		long endTime= System.currentTimeMillis();
		System.out.println("task finished!");
		System.out.println("time consumed = " + (endTime - startTime)/1000.0 + " seconds");
	}

}
