/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
public class KnnReg {

    private static final long serialVersionUID = 1L;

    public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

    public IntOption limitOption = new IntOption("limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

    public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
            "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                    "KDTree search algorithm for nearest neighbour search"
            }, 0);





    public String getPurposeString() {
        return "kNN: special.";
        
    }

    protected Instances window;


    public void setModelContext(InstancesHeader context) {
        try {
            this.window = new Instances(context, 0); //new StringReader(context.toString())
            this.window.setClassIndex(context.classIndex());
        } catch (Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }


    public void resetLearningImpl() {
        this.window = null;
    }


    public void trainOnInstanceImpl(Instance inst) {

        if (this.window == null) {
            this.window = new Instances(inst.dataset());
        }

        if (this.limitOption.getValue() <= this.window.numInstances()) {
            this.window.delete(0);
        }
        this.window.add(inst);
    }



    public double getMeanForInstance(Instance inst) {
        double mean = 0;
        double sum = 0 ;
        try {
            NearestNeighbourSearch search;
            if (this.nearestNeighbourSearchOption.getChosenIndex() == 0) {
                search = new LinearNNSearch(this.window);
            } else {
                search = new KDTree();
                search.setInstances(this.window);
            }
            if (this.window.numInstances() > 0) {
                Instances neighbours = search.kNearestNeighbours(inst, Math.min(kOption.getValue(), this.window.numInstances()));
                for (int i = 0; i < neighbours.numInstances(); i++) {

                    sum  = sum + neighbours.instance(i).classValue() ;
                }
                mean = sum / neighbours.numInstances() ;
            }
        } catch (Exception e) {
            //System.err.println("Error: kNN search failed.");
            //e.printStackTrace();
            //System.exit(1);
            return 1;
            
        }

        return mean ;
    }


    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }


    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }

   
}
