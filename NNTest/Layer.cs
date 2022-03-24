using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    class Layer
    {
        public int nrIpnut;
        public int nrOutput;

        public double[] inputs;
        public double[] outputs; //a

        public double[] biasDerivative;
        public double[,] weightDerivative;

        public double[,] weightOmega;

        public double[] biasOmega;

        public double[,] deltaWeight;

        public double[,] learningRate;

        public double[] learningRateBiases;

        public double[] sumVector; // z
        public double[,] weights; //W
        public double[,] prevWeights;
        public double[] biases;
        public double[] biasesCorrection;

        private static Random rnd=new Random();
        
        public Layer(int _in,int _out)
        {
            nrIpnut = _in;
            nrOutput = _out;

            inputs = new double[nrIpnut];
            outputs = new double[nrOutput];
            weights = new double[nrOutput, nrIpnut];

            prevWeights = new double[nrOutput, nrIpnut];

            biases = new double[nrOutput];

            biasesCorrection = new double[nrOutput];

            sumVector = new double[nrOutput];

            biasDerivative = new double[nrOutput];
            weightDerivative = new double[nrOutput, nrIpnut];

            weightOmega= new double[nrOutput, nrIpnut];

            biasOmega= new double[nrOutput];

            deltaWeight = new double[nrOutput, nrIpnut];

            learningRate = new double[nrOutput, nrIpnut];

            learningRateBiases = new double[nrOutput];

            InitWeightsAndBiases();

        }
        private double GetRandomNumber(double minimum, double maximum)
        {
            return rnd.NextDouble() * (maximum - minimum) + minimum;
        }
        private void InitWeightsAndBiases()
        {
            for(int i=0;i<nrOutput;i++)
            {
                biases[i] = GetRandomNumber(-0.5, 0.5);
                biasOmega[i] = 0;
                for (int j=0;j<nrIpnut;j++)
                {
                    learningRate[i, j] = 0.1;
                    weights[i, j] = GetRandomNumber(-0.5, 0.5);
                    prevWeights[i, j] = weights[i, j];
                    deltaWeight[i, j] = 0;
                    weightOmega[i, j] = 0;
                }
            }
        }


        public void multiWeightInput()
        {
            for(int i=0;i<weights.GetLength(0);i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < weights.GetLength(1);j++)
                {                    
                    outputs[i]+= weights[i, j] * inputs[j];                        
                }
                outputs[i] += biases[i];
                sumVector[i] = outputs[i]; 
                outputs[i] = Sigmoid(outputs[i]);
            }
        }
        private double Sigmoid(double x)
        {
            return 1f / (1f + Math.Exp(-x));
        }




    }
}
