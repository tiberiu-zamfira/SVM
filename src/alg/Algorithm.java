package alg;

import svm.SVM;

public abstract class Algorithm  extends Thread{
	SVM svm;
	public String name;  //Numele algoritmului
	public float eta;  //Rata de invatare
	public long P = 10000000;  //Numarul de stagii, S la curs
	public int N;  //Numarul de vectori de invatare
	public int dim;  //Dimensiunea vectorilor de invatare
	public long t;  //Timpul de executie
	
	public Algorithm(SVM svm){
		this.svm = svm;
		eta = svm.settings.learning_rate;
		if(svm.ind.ALL != null){ //Daca avem vectori de invatare incarcati
			N = svm.ind.V.length;
			dim = svm.ind.V[0].getDimension();		
			svm.outd.dataInputFile = svm.ind.input_file;
			svm.outd.vectors_count = N;
			svm.outd.attributes_count = dim;
			svm.outd.max_stages_count = P;	
			svm.control.start.enable(true);	
			svm.control.options.enable(true);
		}
	}

	public void start_simulation(){
		svm.design.calculates = true; //Pentru a face capul sa dispara/apara
		svm.design.repaint();
		start();  //Din thread -> porneste run-ul acestui fir de executie
	}

	public abstract void run();  //De suprascris in subclase, aici se pune codul algoritmului
	
	public int getAccuracy(float[] w){  //Calculeaza acuratetea pentru setul de testare
		int accuracy = 100;
		if(svm.ind.T != null && w != null){
			svm.outd.testing_vectors_count = svm.ind.T.length;
			int hit = 0;
			for(int i = 0; i < svm.ind.T.length; i++){
				float s = 0;
				for(int j = 0; j < dim; j++)
					s += w[j]*svm.ind.T[i].X[j];
				s += w[dim];
				int y = 1;
				if(s < 0) y = 0;
				if(y == svm.ind.T[i].cl.Y) hit++;
			}
			accuracy = (hit*100)/svm.ind.T.length;
		}
		return accuracy;
	}
	
}