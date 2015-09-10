package cs691.assignment03;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.stat.*;
import java.lang.Math;


import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.*;
import weka.core.converters.CSVLoader;
import org.apache.commons.math3.stat.inference.TTest;;

/**
 * Below is a template that you MAY (read: SHOULD) follow to help complete the assignment. If you want to
 * try to complete the assignment in a different way, feel free to explore but I have cut out a lot of the
 * work already if you follow this code. Below I give examples of the overall flow as well as examples of 
 * how to use various Weka objects to complete the assignment.
 * 
 * Read the ENTIRE FILE before making changes. I have marked TODO where there is something you need to
 * change or use later.
 * @author sloscal1
 *
 */
public class Driver2 {

	/**
	 * This program expects to be run like: <br />
	 * <code>java cs691.assignment03.Drive <performance results file name> <comparison results file name> <training data file> </code>
	 *
	 * @param args[0] is the output file name for the performance results
	 *        args[1] is the output file name for the algorithm comparison results
	 *        args[2] is the input dataset file
	 * @throws Exception I have not handled any exceptions in my code. You may want to for debugging purposes.
	 */


	public static void main(String[] args) throws Exception {	
		//Done Construct a weka Instances object representing the data;
		//read the arff file
		BufferedReader file = new BufferedReader(new FileReader(new File(args[2])));
		Instances dataset = new Instances(file);
		// Make the last attribute be the class
		dataset.setClassIndex(dataset.numAttributes() - 1);
		//Set the number of runs
		int numRuns = 10;
		//DoneTODO Set the random seed for repeatability (PICK AN ARBITRARY SEED THAT IS DIFFERENT THAN THIS ONE!)
		long randSeed = 1767532L;
		//Set up your output files:
		PrintWriter perfResults = new PrintWriter(new BufferedWriter(new FileWriter(new File(args[0]))));
		PrintWriter compResults = new PrintWriter(new BufferedWriter(new FileWriter(new File(args[1]))));
		// you will want to print some header information to help track
		//the results in the output files. That can be accomplished where you instantiated
		//the classifiers, or later in the experimental loop.

		//Done Instantiate the classifiers and set the appropriate parameters given in the
		//assignment
		IBk knn1 = new IBk(1);
		IBk knn3 = new IBk(3);
		IBk knn5 = new IBk(5);
		IBk knn10 = new IBk(10);

		J48 tree = new J48();
		tree.setMinNumObj(2);
		J48 tree2 = new J48();
		tree2.setMinNumObj(5);
		J48 tree3 = new J48();
		tree3.setMinNumObj(10);
		J48 tree4 = new J48();
		tree4.setMinNumObj(30);

		NaiveBayes nBayes = new NaiveBayes();
 
		Kernel poly = new PolyKernel();
		Kernel rbf = new RBFKernel();
		SMO smo1 = new SMO();
		smo1.setC(1.0);
		smo1.setKernel(poly);
		SMO smo2 = new SMO();
		smo2.setC(5.0);
		smo2.setKernel(poly);
		SMO smo3 = new SMO(); 
		smo3.setC(1.0);
		smo3.setKernel(rbf);
		SMO smo4 = new SMO();
		smo4.setC(5.0);
		smo4.setKernel(rbf);

		//...

		//Done Add them all to a list of classifiers
		List<Classifier> classifiers = new ArrayList<>();
		classifiers.add(knn1);
		classifiers.add(knn3);
		classifiers.add(knn5);
		classifiers.add(knn10);
		classifiers.add(tree);
		classifiers.add(tree2);
		classifiers.add(tree3);
		classifiers.add(tree4);
		classifiers.add(nBayes);
		classifiers.add(smo1);
		classifiers.add(smo2);
		classifiers.add(smo3);
		classifiers.add(smo4);
		//...

		//DoneTODO You can generate the results in a variety of ways. I have
		//given the template for an object-oriented way below. If you are
		//uncomfortable with this, consider making several different loops
		//(one for each training data generation method) and doing the data set
		//construction process within each of those loops.
		//IMPLEMENETATION NOTE: The way I have completed is not a particularly memory efficient way of
		//doing things. Instead, you might want to keep track of lists of instance id's
		//(instead of full copies of each instance) and create the Instances objects right
		//where you use them and get rid of them immediately after. NOT REQUIRED for the
		//assignment.

		List<Generator> tdGenerators = new ArrayList<>();
		tdGenerators.add(new HoldoutGenerator(dataset, randSeed, 1.0/3.0));
		tdGenerators.add(new CrossValidationGenerator(dataset, randSeed));
		tdGenerators.add(new LOOCVGenerator(dataset, randSeed));
		tdGenerators.add(new ResampleGenerator(dataset, randSeed, 1.0/3.0));
		//DoneTODO add the other generators using the parameters specified in the assignment...

		//Make a place to keep track of the results before they are printed
		Map<String, double[]> performance = new LinkedHashMap<>();
		NameGen ng = new NameGen();
		
		//This is the main testing loop.
		//For each classification algorithm and parameter setting...
		for(Classifier alg : classifiers){
			//Break up the data according to each of the methods, i.e., holdout, cross validation, etc.
			for(Generator g : tdGenerators){
				//DoneTODO Each iteration of this loop creates data for 2 columns of the performance results file (resub and generalization)
				//Store that information:
				double[] resubErrors = new double[numRuns];
				double[] genErrors = new double[numRuns];
				for(int run = 0; run < numRuns; ++run){
					g.initializeRun();
					double avgPartResubErr = 0.0;
					double avgPartGenErr = 0.0;
					for(int part = 0; part < g.getNumPartitions(); ++part){
						Instances train = g.getNextTrainingSet(part);
						System.out.println("Run: " + run + " Part: " + part);
						System.out.println("Number of train instances = " + train.numInstances());
						Instances test = g.getNextTestingSet(part);
						System.out.println("Number of test instances = " + test.numInstances());
						alg.buildClassifier(train);
						//DoneTODO Get the resubstitution error and estimated generalization error 
						//of the classifier alg.
						double resubErr = 0.0;
						double genErr = 0.0;
						for(int i = 0; i < train.numInstances(); i++){
							double trueValue, predictValue;
							trueValue = train.instance(i).classValue();
							predictValue = alg.classifyInstance(train.instance(i));
							if(trueValue != predictValue){
								resubErr++;
							}
						}
						for(int i = 0; i < test.numInstances(); i++){
							double trueValue, predictValue;
							trueValue = test.instance(i).classValue();
							predictValue = alg.classifyInstance(test.instance(i));
							if(trueValue != predictValue){
								genErr++;
							}
						}
						resubErr=resubErr/(train.numInstances()+test.numInstances());
						genErr=genErr/(train.numInstances()+test.numInstances());
						//DoneTODO HINT: the classifyInstance method gets a predicted class label
						//from alg, and it can be compared against the true class label using
						//train.instance(i).classValue() for some instance i
						avgPartResubErr += resubErr;
						avgPartGenErr += genErr;
					}
					//Store the error rates in so they can be printed later
					resubErrors[run] = avgPartResubErr/g.getNumPartitions();
					genErrors[run] = avgPartGenErr/g.getNumPartitions();
					//IMPORTANT: reset the random generator so that the EXACT SAME training
					//and testing sets can be created for all algorithms in the test.
					g.reset();
				}
				//DoneTODO Make an appropriate key to keep track of each algorithm's results.
				//You can make these up when writing the header information, like:
				//algName-paramValue-trainingGenerationMethod, so the first one would be
				//IBk-1-Holdout, and the second line would be IBk-1-Holdout-gen
				//Using alg.toString() gives unpredictable results since different Weka algorithms
				//implement toString differently. It is just used as a placeholder!
				
				String str = ng.getName(alg);
				System.out.println(str+g.toString()+ " Classification done!");
				performance.put(str+g.toString(), resubErrors);
				performance.put(str+g.toString()+"gen", genErrors);
			}
		}

		//DoneTODO Print the performance results in the appropriate order for the performance file.
		//Also compute the mean and standard deviation for each (feel free to use the StatUtils class
		//from the apache commons math library) and print them in the file.

		int classSize = classifiers.size();
		int genSize = tdGenerators.size();
		System.out.println("class size is " + classSize);
		System.out.println("number of generator is "+ genSize);
		System.out.println("All classifications done. Generating performance and comparison files.");

		perfResults.println("#, run1, run2, run3, run4, run5, run6, run7, run8, run9, run10, avg, std");

		for(int i = 0; i<classSize; i++){
			for(int j = 0; j <genSize ; j++){
				String str = ng.getName(classifiers.get(i));
				double resubResult[] = performance.get(str+tdGenerators.get(j).toString());
				double genResult[]= performance.get(str+tdGenerators.get(j).toString()+"gen");
				double meanResubErrors = StatUtils.mean(resubResult);
				double stdResub = Math.pow(StatUtils.variance(resubResult), 0.5);	
				double meanGenErrors = StatUtils.mean(genResult);
				double stdGen = Math.pow(StatUtils.variance(genResult), 0.5);
				perfResults.print(str+tdGenerators.get(j).toString()+",");
				for(int pos = 0; pos < resubResult.length; pos++ ){
					perfResults.print(resubResult[pos]+",");
				}
				perfResults.print(meanResubErrors+",");
				perfResults.println(stdResub);
				perfResults.print(str+tdGenerators.get(j).toString()+"gen"+",");
				for(int pos = 0; pos < genResult.length; pos++ ){
					perfResults.print(genResult[pos]+",");
				}
				perfResults.print(meanGenErrors+",");
				perfResults.println(stdGen);
			}
		}
		System.out.println("Performance file generated");
		double p;
		TTest t = new TTest();
		for(int i = 0; i<classSize; i++){
			for(int j = 0; j <genSize ; j++){
				for(int m = 0; m<classSize; m++){
					for(int n = 0; n <genSize; n++){
						String str = ng.getName(classifiers.get(i));
						String str2 = ng.getName(classifiers.get(m));
						double a[] = performance.get(str+tdGenerators.get(j).toString());
						double b[] = performance.get(str2+tdGenerators.get(n).toString());
						double c[] = performance.get(str2+tdGenerators.get(n).toString()+"gen");
						p = t.pairedTTest(a, b);
						compResults.print(p + ",");
						p = t.pairedTTest(a, c);
						System.out.print(p);
						compResults.print(p + ",");
					}
				}
				compResults.println();
				System.out.println();

				for(int m = 0; m<classSize; m++){
					for(int n = 0; n <genSize; n++){
						String str = ng.getName(classifiers.get(i));
						String str2 = ng.getName(classifiers.get(m));

						double a[] = performance.get(str.toString()+tdGenerators.get(j).toString()+"gen");
						double b[] = performance.get(str2+tdGenerators.get(n).toString());
						double c[] = performance.get(str2+tdGenerators.get(n).toString()+"gen");
						p = t.pairedTTest(a, b);
						compResults.print(p + ",");
						p = t.pairedTTest(a, c);
						compResults.print(p + ",");
					}
				}
				compResults.println();

			}
		}



		//DoneTODO Perform t-tests to determine which, if any, algorithms have a performance
		//advantage on this data set, and what parameter setting works best.
		//Compare every set of generalization performances against all others
		//using the pairedTTest method from org.apache.commons.math3.stat.inference.TTest
		//to obtain a p-value. Record the p-values in the compResults file.


		//Close the output files to ensure that all results have been written.
		perfResults.close();
		compResults.close();
	}

	//DoneTODO The classes below could be in their own files, I just grouped them all
	//together here so I could upload a single file to Blackboard. You can also drop
	//the static modifiers from them if you decide to move them from this file.

	/**
	 * Generator encapsulates the process of creating training and testing data subsets
	 * from a data source in a repeatable fashion.
	 * 
	 * NOTE: this code expects that <code>getNextTrainingSet</code> is ALWAYS
	 * called before <code>getNextTestingSet</code> is called to ensure that they
	 * correspond (i.e., data in the testing set is not included in the training set).
	 * 
	 * @author sloscal1
	 *
	 */
	public static abstract class Generator {
		/** A copy of the source data used to generate training and testing samples */
		protected Instances dataCopy;
		/** A source of randomness */
		protected Random srcRand;
		/** Seed to control the random behavior for experimental repeatability */
		protected long randSeed;

		/**
		 * Create a new training data Generator from the specified source data and
		 * with the given seed for experimental repeatability.
		 * 
		 * @param srcData must not be null
		 * @param randSeed 
		 */
		public Generator(Instances srcData, long randSeed){
			if(srcData.classIndex() <  0)
				srcData.setClassIndex(srcData.numAttributes()-1);
			dataCopy = new Instances(srcData);
			this.randSeed = randSeed;
			srcRand = new Random(randSeed);
		}

		/**
		 * Perform any book-keeping prior to starting a run. Useful to override for the CV method.
		 */
		public void initializeRun(){
			dataCopy.randomize(srcRand);
		}

		/**
		 * Get the next training set generated according to a specific
		 * training data generation method.
		 * 
		 * @return non-null set of data instances drawn from the source data
		 */
		public abstract Instances getNextTrainingSet(int foldNum);

		/**
		 * Get the next testing set generated according to a specific
		 * testing generation method.
		 * @return non-null set of data instances drawn from the source data
		 */
		public abstract Instances getNextTestingSet(int foldNum);

		/**
		 * Return the number of individual partitions that must be calculated for this
		 * training method. MUST be overridden for cross validation and LOOCV.
		 * @return
		 */
		public int getNumPartitions(){
			return 1;
		}

		/**
		 * Reset the random state of this training data Generator.
		 */
		public void reset(){
			srcRand = new Random(randSeed);
		}
	}

	/**
	 * This class implements a random hold-out procedure.
	 * @author sloscal1
	 *
	 */
	public static class HoldoutGenerator extends Generator{
		/** The fraction of the data to randomly hold out for testing. */
		private int trainingSize;

		/**
		 * Create a new training set generator following the Holdout process
		 * @param srcData must not be null
		 * @param randSeed must not be null
		 * @param holdoutFraction must be in the range (0,1)
		 */
		public HoldoutGenerator(Instances srcData, long randSeed, double holdoutFraction) {
			super(srcData, randSeed);
			trainingSize = (int)((1-holdoutFraction)*dataCopy.numInstances());
		}

		@Override
		public Instances getNextTrainingSet(int foldNum) {
			//The randomization is done automatically by initializeRun at the start of each run.
			return new Instances(dataCopy, 0, trainingSize);
		}

		@Override
		public Instances getNextTestingSet(int foldNum) {
			//Assumes that the data has been shuffled around in getNextTrainingSet so we just need
			//to copy over the remaining data
			return new Instances(dataCopy, trainingSize, dataCopy.numInstances() - trainingSize);
		}

		@Override
		public String toString(){
			return "Holdout";
		}
	}
	//DoneTODO IF you want to follow my template, fill in the classes below with the
	//appropriate code to achieve each data generation strategy. It looks like a lot, but
	//it's only a few lines of code for each class.

	public static class CrossValidationGenerator extends Generator{

		private int fold;

		public CrossValidationGenerator(Instances srcData, long randSeed) {
			super(srcData, randSeed);
			fold = 10; // DoneTODO Should keep track of the number of folds
		}

		@Override
		public  void initializeRun() {
			super.initializeRun();
			dataCopy.stratify(fold); //DoneTODO make sure you use stratified sampling
		}

		@Override
		public Instances getNextTrainingSet(int foldNum) {
			Instances train = dataCopy.trainCV(fold, foldNum);
			return train;
		}

		@Override
		public Instances getNextTestingSet(int foldNum) {
			Instances test = dataCopy.testCV(fold, foldNum);
			return test;
		}

		@Override
		public int getNumPartitions() {
			return fold; // DoneTODO Return the number of partitions done by CV
		}

		@Override
		public void reset() {
			super.reset();
			fold = 10; //DoneTODO reset any state variables used to keep track of folds
		}

		@Override
		public String toString(){
			return "CV";
		}
	}
	public static class LOOCVGenerator extends Generator{

		private int fold;
		private Instances dataCopy2;
		private Instances testSet;
		private Instance singleInst;
		public LOOCVGenerator(Instances srcData, long randSeed) {
			super(srcData, randSeed);
			dataCopy2 = new Instances(dataCopy, 0, 100); //copying the first 100 instances
			fold = dataCopy2.numInstances();
			testSet = new Instances(dataCopy2, 0, 1);//finding out the number of folds = 100
			// DoneTODO Very similar to CrossValidation...
		}
		
		@Override
		public  void initializeRun() {
			super.initializeRun();
			//DoneTODO make sure you use stratified sampling
		}

		@Override
		public Instances getNextTrainingSet(int foldNum) {
			dataCopy2 = new Instances(dataCopy, 0, 100);
			singleInst = dataCopy2.get(foldNum);
			dataCopy2.delete(foldNum);//removing the instance at the foldNum position from dataCopy2 and storing it in Instance singleInst
			testSet.set(0, singleInst); //adding the singleInst to a new Instances obj to be used in getNextTestingSet function line 512
			return dataCopy2; // returning the new dataCopy2 without the removed instance DoneTODO Use trainCV
		}

		@Override
		public Instances getNextTestingSet(int foldNum) {
			return testSet; // returning the instance removed from dataCopy2 DoneTODO Use testCV

		}

		@Override
		public int getNumPartitions() {
			return fold; // DoneTODO Return the number of partitions done by LOOCV
		}

		@Override
		public void reset() {
			super.reset();
			fold = 100; //DoneTODO reset any state variables used to keep track of instances
			dataCopy2 = new Instances(dataCopy, 0, 100);
			testSet = new Instances(dataCopy2, 0, 1);
		}

		@Override
		public String toString(){
			return "LOOCV";
		}
	}
	public static class ResampleGenerator extends Generator{

		private int trainingSize;
		Random rGen = new Random();

		public ResampleGenerator(Instances srcData, long randSeed, double trainFraction) {
			super(srcData, randSeed);
			trainingSize = (int)(trainFraction*dataCopy.numInstances());
			// DoneTODO Need to know how big to create the training set
			// If using the Resample class, we need to import it, but after that I have no idea how it works
			// The following is assuming that Resample is the same as Random Subsampling (taught in class)

		}

		@Override
		public  void initializeRun() {
			super.initializeRun();

		}

		@Override
		public Instances getNextTrainingSet(int foldNum) {
			// DoneTODO Consider building both training and testing set in here...
			Instances temp = dataCopy.resample(rGen);
			return new Instances(temp, 0, trainingSize);
		}

		@Override
		public int getNumPartitions() {
			return 20; // DoneTODO Return the number of partitions done by Resample
		}
		@Override
		public Instances getNextTestingSet(int foldNum) {
			// DoneTODO ...and just return the most recently created set
			Instances temp = dataCopy.resample(rGen);
			return new Instances(temp, trainingSize, dataCopy.numInstances() - trainingSize);
		}

		@Override
		public String toString(){
			return "Resample";
		}
	}
	public static class NameGen{
		public String getName(Classifier alg){
			String str = alg.getClass().getSimpleName();
			String realName = null;
			switch (str){
			case "IBk":
				int knn = ((IBk)alg).getKNN();
				realName = str.concat("."+String.valueOf(knn)+".");
				break;
			case "J48":
				int objNum = ((J48)alg).getMinNumObj();
				realName = str.concat("."+String.valueOf(objNum)+".");
				break;
			case "NaiveBayes":
				realName = str.concat(".");
				break;
			case "SMO":
				double c = ((SMO)alg).getC();
				Kernel k = ((SMO)alg).getKernel();
				Kernel poly = new PolyKernel();
				String tmp = str.concat("."+String.valueOf(c));
				if (k.toString() == poly.toString()){
					realName = tmp.concat(".PolyKernel.");
				}
				else{
					realName = tmp.concat(".RBFKernel.");
				}
				break;
			default:
				System.out.println("no classifier name found");
				break;

			}

			return realName;
		}
	}
}