package cloudType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Dataset extends ArrayList<Record>{
	protected static String unknownClassTitle;
	private Map<String, Integer> classes;
	private String name;
	
	public boolean add(Record record) {
		return super.add(record);
	}
}
