import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Vector;



public class Driver {
	static int fileNo=1;
	public static void main(String[] args) {
		
		// Experiment 1
		// 		    noGenerators, noPoints, delta, minSD, maxSD, fileName
		generateWithParameters(3, 3000, 0.5, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(3, 3000, 1.0, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(3, 3000, 1.5, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(3, 3000, 2.0, 1, 1, "File"+fileNo+".txt");

		generateWithParameters(5, 5000, 0.5, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(5, 5000, 1.0, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(5, 5000, 1.5, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(5, 5000, 2.0, 1, 1, "File"+fileNo+".txt");

		generateWithParameters(10, 10000, 0.5, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(10, 10000, 1.0, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(10, 10000, 1.5, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(10, 10000, 2.0, 1, 1, "File"+fileNo+".txt");

		// Experiment 2
		generateWithParameters(5, 5000, 3, 1, 1, "File"+fileNo+".txt");
		generateWithParameters(5, 5000, 3, 2, 2, "File"+fileNo+".txt");
		generateWithParameters(5, 5000, 3, 3, 3, "File"+fileNo+".txt");

		// Experiment 3
		generateWithParameters(5, 5000, 1.25, 0.75, 2, "File"+fileNo+".txt");

		// Experiment 4
		generateWithParameters(5, 500, 1.25, 0.75, 0.75, "File"+fileNo+".txt");
		generateWithParameters(5, 5000, 1.25, 0.75, 0.75, "File"+fileNo+".txt");
		generateWithParameters(5, 25000, 1.25, 0.75, 0.75, "File"+fileNo+".txt");
		
	} 

	public static void generateWithParameters(int noOfGenerators,int noOfPoints,double delta,double minStDev,double maxStDev,String fileName)
	{
		fileNo++;
		Vector<GaussianGenerator> generators = new Vector<GaussianGenerator>();
		String s ="";
		Random rand = new Random();// You can seed this Object to allow the generators to be picked in the same order at each run.
		for(int i=0;i<noOfGenerators;i++)
		{
			// 													     mean     ,      standard deviation
			GaussianGenerator g= new GaussianGenerator(1+(i*delta), minStDev+(rand.nextDouble()*(maxStDev-minStDev)));
			generators.add(g);
			s+=g+"\n";
		}


		for(int i=0;i<noOfPoints;i++)
		{
			int chosenGenerator=rand.nextInt(generators.size());
			GaussianGenerator g = generators.get(chosenGenerator);
			s+= g.mean+"\t"+g.nextDouble()+"\n";
		}
		System.out.println(s);
		write(fileName, s);

	}

	public static void write(String fileName,String s)
	{
		try
		{ 
			System.out.println(fileName);
			File file = new File(fileName);

			// if file doesnt exists, then create it
			if (!file.exists()) {
				file.createNewFile();
			}

			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			bw.write(s);
			bw.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
