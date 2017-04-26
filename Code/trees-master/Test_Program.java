import java.io.*;
import java.util.ArrayList;
import java.util.Random;

class Pair_{
		Integer first, second;
		Pair_(Integer first,Integer second){
			this.first= first;
			this.second= second;
		}
}

public class Test_Program extends  Thread implements Runnable{
    public static LogicalOrderingAVL<Integer,Integer> avl_tree;
    public static ArrayList<Pair_> query_arr;
    public static int query_size;
    public static int half;
    public static int three_4th;
    public int id;
    @Override
    public void run(){
        for(int i=id;i<query_size;i+=128) {
            if (i < query_size) {
                if (i < half)
                    avl_tree.containsKey(query_arr.get(i).first);
                else if(i<three_4th) avl_tree.put(query_arr.get(i).first,query_arr.get(i).second);
                else avl_tree.remove(query_arr.get(i).first);
            }
        }
    }

    public Test_Program(int id){
        this.id= id;
    }

    public Test_Program(){}

	public static void main(String[] args) {
        avl_tree = new
                LogicalOrderingAVL<>(Integer.MIN_VALUE, Integer.MAX_VALUE);
        String filename = args[0];
        query_size= Integer.parseInt(args[1]);
        half= (query_size>>1);
        three_4th= (three_4th/4)*3;

        long exec_time=0;

            try {
                BufferedReader bf = new BufferedReader(new FileReader(filename));
                String str;
                Integer size = Integer.parseInt(bf.readLine());
                while ((str = bf.readLine()) != null) {
                    String split_str[] = str.split(" ");
                    avl_tree.put(Integer.parseInt(split_str[0]),
                            Integer.parseInt(split_str[1]));
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            int num_threads = 128;
            Test_Program test_program = new Test_Program();
            query_arr= test_program.generate_queries(query_size);
            Test_Program threads[] = new Test_Program[num_threads];
            long start = System.nanoTime();
            for (int i = 0; i < num_threads; ++i) {
                threads[i] = new Test_Program(i);
                threads[i].start();
            }
            try {
                for (int i = 0; i < num_threads; ++i)
                    threads[i].join();
            } catch (InterruptedException ex) {
                ex.printStackTrace();
            }
            exec_time = (System.nanoTime() - start);
        System.out.printf("%.6f\n",(exec_time/5)/1e6);
    }

    public ArrayList<Pair_> generate_queries(int max_query) {
        int max_value= (max_query<<4);
        int min_value = -max_value;
        int range= (max_value - min_value +1);
        Random random=new Random();
        ArrayList<Pair_> temp_arr= new ArrayList<>();
        for(int i=0;i<max_query;++i){
            int key= min_value + random.nextInt(range);
            int val= min_value + random.nextInt(range);
            temp_arr.add(new Pair_(key,val));
        }
        return temp_arr;
    }
}