package moa.classifiers.lazy;
import java.io.StringReader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import moa.classifiers.Regressor;


public class kNNReg extends AbstractClassifier implements Regressor {
    private static final long serialVersionUID = 1L;

    public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

    public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

    public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
            "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                    "KDTree search algorithm for nearest neighbour search"
            }, 0);


    int C = 0;

    @Override
    public String getPurposeString() {
        return "kNN: special.";
    }

    protected Instances window;

    @Override
    public void setModelContext(InstancesHeader context) {
        try {
            this.window = new Instances(context,0); //new StringReader(context.toString())
            this.window.setClassIndex(context.classIndex());
        } catch(Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void resetLearningImpl() {
        this.window = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (inst.classValue() > C)
            C = (int)inst.classValue();
        if (this.window == null) {
            this.window = new Instances(inst.dataset());
        }
        if (this.limitOption.getValue() <= this.window.numInstances()) {
            this.window.delete(0);
        }
        this.window.add(inst);
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double sum = 0 ;
        double prediction = 0 ;
        try {
            NearestNeighbourSearch search;
            if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
                search = new LinearNNSearch(this.window);
            } else {
                search = new KDTree();
                search.setInstances(this.window);
            }
            if (this.window.numInstances()>0) {
                Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
                for(int i = 0; i < neighbours.numInstances(); i++) {

                    sum = neighbours.instance(i).classValue() ;
                    prediction = sum/neighbours.numInstances() ;
                }
            }
        } catch(Exception e) {
            //System.err.println("Error: kNN search failed.");
            //e.printStackTrace();
            //System.exit(1);
            return new double[inst.numClasses()];
        }
        return new double[] {prediction};
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }
}
