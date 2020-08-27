#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

class Event{
public:
	int name;
	int occurence;
	int start;
	int end;

	Event(int n_name, int n_start, int n_end){
		name = n_name;
		start = n_start;
		end = n_end;
	}

	char get_itr(Event other){
		int a1 = start;
		int a2 = end;
		int b1 = other.start;
		int b2 = other.end;

		//before
		if (a2 < b1)
			return 0;//'b';

		//meets
		if (a2 == b1)
			return 1;//'m';

		//overlaps
		if (a1 < b1 and a2 < b2 and b1 < a2)
			return 2;//'o';

		//during
		if (a1 < b1 and b2 < a2)
			return 3;//'d';

		//finishes
		if (b1 < a1 and a2 == b2)
			return 4;//'f';

		//starts
		if (a1 == b1 and a2 < b2)
			return 5;//'s';

		//equals
		return 6;//'e';
	}
};

bool compareEvents(Event e1, Event e2) 
{ 
    if(e1.start < e2.start)
		return true;
    return e1.start == e2.start and e1.end < e2.end;
} 

vector<Event> read_sparse_matrix(string filename, int& num_features){
	vector<Event> event_list;
	
	//open file
	ifstream file (filename, ios::in | ios::binary);

	int num_f;
	//get number of features
	if (file.is_open())
	    file.read ((char*)&num_f, sizeof(num_f));
	num_features = num_f;

	if(num_features > 1000)
		exit(-1);

	//parse the rest of teh file
	int current_feature = -1;
	int p1, p2;

	while (file.peek()!=EOF)
	{
	    file.read ((char*)&p1, sizeof(p1));
	    file.read ((char*)&p2, sizeof(p2));

	    if( p2 == 0 ){
	    	current_feature = p1;
	    }else{
		    Event e = Event(current_feature, p1, p2);
		    event_list.push_back(e);
		}
	}
	return event_list;
}

/*
I need to read and parse the file without using numpy. That way I can completely 
remove the work I need to do with Boost Python. 
*/

bool extract_itr_seq_into_counts(string input_filename, string output_filename){

	// get events from file
	int num_features;
	vector<Event> events = read_sparse_matrix(input_filename, num_features);
	sort(events.begin(), events.end(), compareEvents);

	// get a list of all of the ITRs in the txt_file
	int itr_list[num_features][num_features][7];
	fill(
		&itr_list[0][0][0],
   		&itr_list[0][0][0] + sizeof(itr_list) / sizeof(itr_list[0][0][0]),
   		0);

	for (int i = 0; i < events.size(); i++){
		int j = i+1;
		while (j < events.size() and events[i].name != events[j].name){
			int itr_name = events[i].get_itr(events[j]);

			int e1 = events[i].name;
			int e2 = events[j].name;

			itr_list[e1][e2][itr_name] += 1;

			j += 1;
		}
	}

	// write to output file
	ofstream ofile (output_filename, ios::out | ios::binary);
	int i = 0;
	int j = 0;
	for (int k = 0; k < 7; k++){
		cout << k << ": " << itr_list[i][j][k] << endl;
		//ofile << itr_list[i][j][k];
		ofile.write((char*)&itr_list[i][j][k],sizeof(int));
		//ofile.write((char*)itr_list[i][j][k],sizeof(itr_list[i][j][k]));
	}
	/*
	for (int i = 0; i < num_features; i++){
		for (int j = 0; j < num_features; j++){
			for (int k = 0; k < 7; k++){
				cout << itr_list[i][j][k];
				//ofile << itr_list[i][j][k];
				ofile.write((char*)&itr_list[i][j][k],sizeof(int));
			}
		}
	}
	*/
	ofile.close();

	return 1;
}


int main(int argc, char** argv){
	//get input filename
	string input_filename = argv[1];
	string output_filename = argv[2];

	//run code (open file)
	extract_itr_seq_into_counts(input_filename, output_filename);
}