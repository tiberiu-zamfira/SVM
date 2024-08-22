package alg;

import svm.SVM;

public class Perceptron extends Algorithm{

	public Perceptron(SVM svm){
		super(svm);
		if(svm.ind.V != null){
			name = "Perceptron";
			svm.outd.algorithm = name;
			svm.outd.showInputData();
		}
	}
	
	public void run(){
		t = System.currentTimeMillis();
		boolean flag = false;
		float[] w = new float[dim+1];
		float[] w1 = new float[dim+1];
		for(int j = 0; j < w.length; j++) w1[j] = w[j] = -0.5f + (float)Math.random();
		for(long p = 1; p <= P; p++){
			boolean erori = false;
			for(int i = 0; i < N; i++) {
				float s = 0;
				for(int j = 0; j < dim; j++)
					s += w[j]*svm.ind.V[i].X[j];
				s += w[dim];
				int y = s < 0 ? 0 : 1;
				int e = svm.ind.V[i].cl.Y - y;
				if(e != 0){
					erori = true;
					for(int j = 0; j < dim; j++)
						w[j] += eta*svm.ind.V[i].X[j]*e;
					w[dim] += eta*e;					
				}
			}
			if(!erori) {
				svm.outd.stages_count = p;
				svm.outd.computing_time = System.currentTimeMillis() - t;
				svm.outd.w = w;
				svm.outd.accuracy = getAccuracy(w);
				svm.outd.showInputData();
				svm.outd.showOutputData();
				svm.design.calculates = false;
				svm.design.repaint();
				flag = true;
				break;
			}
		}
		if(!flag) 
			System.out.println(P + " stages have passed. Increase the number of stages and reloaded.");		
		else{		
			for(int j = 0; j < w.length; j++) w[j] = w1[j];
			for(int p = 1; p <= P; p++){
				boolean erori = false;
				for(int i = 0; i < N; i++) {
					float s = 0;
					for(int j = 0; j < dim; j++)
						s += w[j]*svm.ind.V[i].X[j];
					s += w[dim];
					int y = s < 0 ? 0 : 1;
					int e = svm.ind.V[i].cl.Y - y;
					if(e != 0){
						erori = true;
						for(int j = 0; j < dim; j++)
							w[j] += eta*svm.ind.V[i].X[j]*e;
						w[dim] += eta*e;					
					}
				}
				svm.control.ta.append("Stage " + p + "\n");
				String s = "";
				for(int j = 0; j < w.length; j++) s += "w["+j+"] = " + w[j] + "; ";
				svm.control.ta.append(s + "\n");
				
				if(dim==2) svm.design.setPointsOfLine(w);
				try{Thread.sleep(250);}
				catch(InterruptedException ex){}			
				
				if(!erori) {
					svm.outd.w = w;
					svm.outd.accuracy = getAccuracy(w);
					svm.outd.showInputData();
					svm.outd.showOutputData();
					svm.control.start.setLabel("Start Simulation");
					svm.design.repaint();
					break;
				}
			}
		}
		svm.control.start.enable(false);		
	}
	
}