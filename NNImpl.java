/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes = null;//list of the input layer nodes
	public ArrayList<Node> hiddenNodes = null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes = null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet = null;//the training set

	Double learningRate = 1.0; // variable to store the learning rate
	int maxEpoch = 1; // variable to store the maximum number of epochs


	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;

		//input layer nodes
		inputNodes = new ArrayList<Node>();
		int inputNodeCount = trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();

		for(int i = 0; i < inputNodeCount; i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes = new ArrayList<Node> ();
		for(int i = 0; i < hiddenNodeCount; i++)
		{
			Node node = new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j = 0; j < inputNodes.size(); j++)
			{
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes = new ArrayList<Node> ();
		for(int i = 0; i < outputNodeCount; i++)
		{
			Node node = new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j = 0; j < hiddenNodes.size(); j++)
			{
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}

	/**
	 * Get the output from the neural network for a single instance
	 * 
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */

	public int calculateOutputForInstance(Instance inst)
	{
		//set inputs
		int currNode = 0;
		for (Double attrib: inst.attributes) {
			inputNodes.get(currNode).setInput(attrib);
			currNode++;
		}

		//forward propgation
		for (int i = 0; i < hiddenNodes.size(); i++) {
			hiddenNodes.get(i).calculateOutput();
		}
		for (int j = 0; j < outputNodes.size(); j++) {
			outputNodes.get(j).calculateOutput();
		}


		double maxOutput = 0.0;
		int index = 0;
		for (Node out: outputNodes) {
		//	System.out.print(out.getOutput() + ",   ");
			if (out.getOutput() >= maxOutput) {
				maxOutput = out.getOutput();
				index = outputNodes.indexOf(out);
			}
		}
		//System.out.println("classifying as " + index);
		return index;
	}





	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		for (int i = 0; i < maxEpoch; i++) {

			for (Instance ex: trainingSet) {

				//set input values of input nodes to match example's attributes
				int currNode = 0;
				for (Double attrib: ex.attributes) {
					inputNodes.get(currNode).setInput(attrib);
					currNode++;
				}

				//FORWARD PROPAGATION
				for (int j = 0; j < hiddenNodes.size(); j++) {
					hiddenNodes.get(j).calculateOutput();
				}
				for (int l = 0; l < outputNodes.size(); l++) {
					outputNodes.get(l).calculateOutput();
				}



				double[][] deltaVals = new double[hiddenNodes.size()][outputNodes.size()];
				//CALCULATING DELTA W(J, K) VALUES FOR EACH OUTPUT NODE
				for (int v = 0; v < outputNodes.size(); v++) {
					Node currentNode = outputNodes.get(v);
					double outputErr = (ex.classValues.get(v) - currentNode.getOutput()); //(Tk - Ok)
					int gPrimeinK = gPrime(currentNode.getSum());  //G'(inK)

					for (int p = 0; p < currentNode.parents.size(); p++) {
						NodeWeightPair nwp = currentNode.parents.get(p);
						double delta = 0.0;
						Node parentNode = nwp.node;
						double aj = parentNode.getOutput();

						delta = learningRate * aj * outputErr * gPrimeinK;
						deltaVals[p][v] = delta; //change in weight from parent node P to output node V
						//currentNode.addDelta(delta);
						//System.out.println("Added delta = " + delta + " for a " + currentNode.type + " node");
					}
				}

				double[][] inputDeltas = new double[inputNodes.size()][hiddenNodes.size()];
				//CALCULATING DELTA W(I, J) VALUES FOR EACH HIDDEN NODE
				for (int m = 0; m < hiddenNodes.size() - 1; m++) {
					Node currentNode = hiddenNodes.get(m);
					double gPrimeinj = gPrime(currentNode.getSum());
					double summation = 0.0;

					for (int k = 0; k < outputNodes.size(); k++) {	
						Node outNode = outputNodes.get(k);
						double outputErr = (ex.classValues.get(k) - outNode.getOutput()); //(Tk - Ok)
						int gPrimek = gPrime(outNode.getSum());	// G'(ink)
						double weightjk = 0.0;

						for (int p = 0; p < outNode.parents.size(); p++) {
							NodeWeightPair pair = outNode.parents.get(p);
							if (pair.node.equal(currentNode)) {
								//System.out.println("Found correct parent weight");
								weightjk = pair.weight;
								break;
							}
						}
						summation = summation + (weightjk * outputErr * gPrimek);
					}
					
					for (int p = 0; p < currentNode.parents.size(); p++) {
						NodeWeightPair nwp = currentNode.parents.get(p);
						double ai = nwp.node.getOutput();

						double delta = learningRate * ai * gPrimeinj * summation; 
						//change in weight in connection from input node p? to hidden node m
						inputDeltas[p][m] = delta;//??
						
						
						//currentNode.addDelta(delta);
					}

				}


				//UPDATE THE WEIGHTS WITH NEW DELTA VALUES
				for (int n = 0; n < outputNodes.size(); n++) {
					Node node = outputNodes.get(n);
					int currParent = 0;
					for (int p = 0; p < node.parents.size(); p++) {
						NodeWeightPair nwp = node.parents.get(p);
						//nwp.weight = nwp.weight + node.getDelta(currParent); 
						nwp.weight = nwp.weight + deltaVals[p][n];
						currParent++;
					}
				}

				for (int h = 0; h < hiddenNodes.size() - 1; h++) {
					Node hiddenNode = hiddenNodes.get(h);
					int currParent = 0;
					for (int p = 0; p < hiddenNode.parents.size(); p++) {
						NodeWeightPair nwp = hiddenNode.parents.get(p);
						//nwp.weight = nwp.weight + hiddenNode.getDelta(currParent);
						nwp.weight = nwp.weight + inputDeltas[p][h];
						currParent++;
					}
				}
			}
		}
	}

	public int gPrime(double input) {
		if (input > 0) return 1;
		return 0;
	}
}
