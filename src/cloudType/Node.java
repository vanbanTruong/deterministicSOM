package cloudType;

import java.util.Collection;
import java.util.LinkedList;

public class Node {
	
	public int col;
	public int row;
	public double[] prototype;
	
	protected Collection<Record> records; // Features associated with this node
	private Record rep; // One particular feature, and the distance between this feature and its bmu
	private double repDist;
	
	public Node(int row, int col, int size) {
		this.row= row;
		this.col= col;
		this.prototype= new double[size];
		initPrototype(0);
		this.records= new LinkedList<Record>();
		this.rep= null;
		this.repDist= Double.MAX_VALUE; 
	}
	
	/**
	 * @param record a record that this node is the best match unit for
	 */
	public void add(Record record) {
		double recordDist= distance(record.getFeatures());
		if (rep== null || recordDist< repDist) {
			rep= record;
			repDist= recordDist;
		}
		records.add(record);
		record.bmu= this;
	}
	
	public void remove(Record record) {
		records.remove(record);
		if (record== rep) {
			refreshRep();
		}
	}
	
	private void refreshRep() {
		for (Record record : records) {
			double recordDist= distance(record.getFeatures());
			if (rep== null || recordDist< repDist) {
				rep= record;
				repDist= recordDist;
			}
		}
	}
	
	/**
	 * Compute the Euclidean distance between this node
	 * and the given feature vector.
	 * @param features a set of features
	 * @return the distance
	 */
	public double distance(double[] features) {
		double sum= 0;
		for (int i = 0; i < prototype.length; i++) {
			sum+= Math.pow(prototype[i]- features[i], 2);
		}
		return sum;
	}
	
	protected void initPrototype(double v) {
		for (int i = 0; i < prototype.length; i++) {
			prototype[i]= v;
		}
	}
	
	protected double[] getPrototype() {
		return this.prototype;
	}
	/**
	 * 
	 * @param features
	 * @param learningRate
	 * @param influence
	 * @return amount of update
	 */
	public double update(double[] features, double learningRate, double influence) {
		double change= Double.MAX_VALUE;
		for (int i = 0; i < prototype.length; i++) {
			change= learningRate* influence* (features[i]- prototype[i]);
			prototype[i]= prototype[i]+ change;
		}
		
		return change;
	}
	
}
