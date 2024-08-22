package alg;

import svm.SVM;

public class Perceptron_Optim extends Algorithm{

	public Perceptron_Optim(SVM svm){
		super(svm);
		if(svm.ind.V != null){
			name = "Perceptron_Optim";
			svm.outd.algorithm = name;
			svm.outd.showInputData();
		}
	}
	
	public void run(){
		t = System.currentTimeMillis();
		boolean flag = false;
		float[] w = new float[dim+1];
		float[] w1 = new float[dim+1];
		float[] M0 = new float[dim];
		float[] M1 = new float[dim];
		int k0 = 0, k1 = 0;		
		for(int i = 0; i < N; i++)
			if(svm.ind.V[i].cl.Y == 0) k0++; else k1++;
		for(int j = 0; j < dim; j++)
			for(int i = 0; i < N; i++)
				if(svm.ind.V[i].cl.Y == 0) 
					M0[j] += svm.ind.V[i].X[j];
				else 
					M1[j] += svm.ind.V[i].X[j];
		for(int j = 0; j < dim; j++){
			M0[j] /= k0;
			M1[j] /= k1;
		}
		float[] X0 = new float[dim];
		for(int j = 0; j < dim; j++){
			X0[j] = (M0[j] + M1[j])/2;
			w[j] =  M1[j] - M0[j];
			w[dim] -= w[j] * X0[j];
		}

		for(int j=0; j<=dim; j++)
		{
			w1[j] = w[j];
		}

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
						{
						w[j] += eta*svm.ind.V[i].X[j]*e; 
						//w[j] += eta*(svm.ind.V[i].X[j] -10 + 10*(float)Math.random())*e; //~200ms
						//w[j] += eta*(svm.ind.V[i].X[j] -20 + 20*(float)Math.random())*e; // + Median => t < 30-40%, s < ~20%
						
						}
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