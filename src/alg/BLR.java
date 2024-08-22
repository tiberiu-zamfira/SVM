//Binomial Logistic Regression (BLR)
package alg;

import svm.SVM;

public class BLR extends Algorithm
{
    public BLR(SVM svm){
		super(svm);
		if(svm.ind.V != null){
			name = "Binomial Logistic Regression";
			svm.outd.algorithm = name;
			svm.outd.showInputData();
		}
	}

    public void run()
	{
		t = System.currentTimeMillis();

		float[] w = new float[dim+1];
		float[] w_old = new float[dim+1];
		for(int j = 0; j < w.length; j++) 
			w[j] = -0.5f + (float)Math.random(); //w[dim] = b
		
		//Maximizarea MLE pentru functia logL
		svm.design.calculates = false;
		if(dim>2)
			svm.design.calculates = true;
		boolean flag = false;
		int nr_it = 100;
		for(int t=1; t<=nr_it; t++)
				for(int i=0; i<N; i++)
					for(int j=0; j<dim; j++)
					{
						for(int k=0; k<dim+1; k++)
							w_old[k] = w[k];
						w[j] -= eta * 
								(-svm.ind.V[i].X[j] * (svm.ind.V[i].cl.Y - p_i(svm.ind.V[i].X, w_old))) / 
								(-1 * p_i(svm.ind.V[i].X,  w_old) * (1 - p_i(svm.ind.V[i].X, w_old) ) * svm.ind.V[i].X[j] * svm.ind.V[i].X[j]);
						
						if(dim==2) svm.design.setPointsOfLine(w);
						System.out.println("w[" + j + "]= " + w[j]); //Verificare in log 
					}
		
					svm.outd.stages_count = 1;
					svm.outd.max_stages_count = 1;
					svm.outd.computing_time = System.currentTimeMillis() - t;
					svm.outd.w = w;
					svm.outd.accuracy = getAccuracy(w);
					svm.outd.showInputData();
					svm.outd.showOutputData();
					svm.design.calculates = false;
					svm.design.repaint();
					svm.control.start.enable(true);		
	}
	
	public float p_i(float[] X, float[] w)
	{
		float s=0;
		for(int i=0; i<dim; i++)
			s += X[i]*w[i];
		s += w[dim];
		return (float)(1 / (1 + (float)Math.exp(-s))+0.01); //Pentru a evita eraorea NaN cand numitorul tinde la 0
	}

}