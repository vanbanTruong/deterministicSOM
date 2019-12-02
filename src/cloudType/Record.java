package cloudType;

import weka.core.Instance;

public class Record {
	private String className;
	Instance featureVector;
	public Node bmu= null;
	
	public Record(Instance featureVector) {
		this.featureVector= featureVector;
	}
	
	/**
	 * Returns the class name of this record.
	 * @return the class name or null if the class is unknown
	 */
	public String getClassName() {
		if (className== null) {
			return Dataset.unknownClassTitle;
		} else {
			return className;
		}
	}
	
	public double[] getFeatures() {
		return this.featureVector.toDoubleArray();
	}
}
