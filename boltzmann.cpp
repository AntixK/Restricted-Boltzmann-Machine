#include <ctime>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <iostream>
#include <sys/stat.h>

#define TRUE 1
#define FALSE 0
#define DEBUG 1
#define FILE 1

using namespace std;

/* Custom Data type names */
typedef std::vector<bool> vect_bool;
typedef std::vector<double> vect_double;
typedef std::vector<vector<bool> > matrix_bool;
typedef std::vector<vector<double> > matrix_double;

/* Structure for Random Number Generation */
struct random
{
    double standard_deviation;

    /* Membership Functions */
    void set_random_seed();
    double generate_random(double lower_limit, double higher_limit);

};

/* Restricted Boltzmann Machine Class */
class RBM : public random
{
   private:
        bool net_stat;
        bool ready_to_train;
        uint8_t bias_init_type;

        double learning_rate;

        vect_double error;

        ofstream log_file;

        uint16_t num_hidden;
        uint16_t num_visible;
        uint16_t train_data_rows;
        uint16_t train_data_cols;

        uint32_t curr_epoch;
        uint32_t epochs;

        matrix_bool data;
        matrix_bool pos_hidden_states;

        matrix_double weights;

        matrix_double pos_associations;
        matrix_double neg_associations;

        matrix_double pos_hidden_probs;
        matrix_double neg_visible_probs;
        matrix_double neg_hidden_probs;

        matrix_double pos_hidden_activations;
        matrix_double neg_visible_activations;
        matrix_double neg_hidden_activations;

    public:

        /* Constructor Functions */
        RBM();
        bool Get_netstat();
        bool Get_ready_to_train();
        void Set_netstat(bool data);
        void set_std(double value=0.01);
        void Set_ready_to_train(bool data);

        /* Data Assembling Functions */
        bool Get_data(matrix_bool &arr,uint16_t nrows);

        /* Data Display Functions */
        void Display_data(uint8_t precision = 7, char *notation ="scientific");
        void Display_error(uint8_t precision = 7, char *notation ="scientific");
        void Display_weights(uint8_t precision = 7, char *notation ="scientific");
        void Display_Pos_hidden_activation(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Neg_hidden_activation(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Neg_visible_activation(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Neg_visible_probs(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Neg_hidden_probs(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Pos_hidden_probs(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Neg_associations(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Pos_associations(uint8_t precision=7,
                                           char *notation ="scientific");
        void Display_Pos_hidden_States(uint8_t precision=7,
                                           char *notation ="scientific");

        /* Handle Files */
        bool Create_file();
        bool Check_file(const string &name);

        /* RBM Initialization Functions */
        bool Init_bias(char *type = "zeros");
        bool Init_weights(char *type = "gaussian");
        bool Init_RBM(uint16_t no_hidden, uint16_t no_visible, double alpha);

        /* RBM Parameters Configuration Functions */
        void Config_probs();
        void Config_error();
        void Config_activations();
        void Config_associations();
        void Config_hiddden_states();

        /* RBM core Functions */
        void Update_error();
        void Compute_error();
        void Set_data_bias();
        void Update_weights();
        void Mat_mul(bool course); /*{ 1= row-wise; 0 = column-wise } */
        double Logistic(double value);
        void Compute_neg_associations();
        void Compute_pos_associations();
		void Compute_probs(uint8_t flag);
        void Compute_pos_hidden_states();
		void Compute_pos_visible_states();
		void Set_neg_visible_probs_bias();
        void Compute_pos_hidden_activations();
        void Compute_neg_hidden_activations();
        void Compute_neg_visible_activations();
        bool RBM_train(uint32_t epochs = 3000, bool method=FALSE);
};

void random::set_random_seed()
{
    srand (static_cast <unsigned> (time(0)));
}

double random::generate_random(double lower_limit, double higher_limit)
{
   return(lower_limit + static_cast <float> (rand())
        /( static_cast <float> (RAND_MAX/(higher_limit-lower_limit))));
}

void RBM::set_std(double value)
{
    standard_deviation = value;
}

RBM::RBM()
{
    cout<<"\n Initializing Restricted Boltzmann Machine ... Success\n";

    num_hidden = 0;
    num_visible = 0;
    learning_rate = 0.1;
    curr_epoch=0;
    epochs=0;

    error.reserve(1);

    weights.reserve(1);
    data.reserve(1);

    pos_hidden_probs.reserve(1);
    pos_associations.reserve(1);
    pos_hidden_states.reserve(1);
    pos_hidden_activations.reserve(1);

    neg_associations.reserve(1);
    neg_hidden_probs.reserve(1);
    neg_visible_probs.reserve(1);
    neg_hidden_activations.reserve(1);
    neg_visible_activations.reserve(1);

    bias_init_type=0;

    set_random_seed();
    set_std();

    #if FILE
        Create_file();
    #endif // FILE
}

void RBM::Config_error()
{
    error.resize(epochs);

    #if DEBUG
        cout<<"\n Error dimensions: 1 * "<<error.size()
            <<"\n" ;
    #endif // DEBUG

    #if FILE
        log_file.open("RBM_Log_File.txt",ios::app);
        log_file<<"\n Error dimensions: 1 * "<<error.size()
                <<"\n" ;
    #endif // FILE
}

bool RBM::Create_file()
{
    if(Check_file("RBM_Log_File.txt"))
    {
        remove("RBM_Log_File.txt");
        #if DEBUG
            cout<<"\n Previous log file found and replaced";
        #endif // DEBUG
    }

    log_file.open("RBM_Log_File.txt");

    if(!log_file.is_open())
    {
        #if DEBUG
            cout<<"\n Error: File not created \n";
        #endif // DEBUG
        return FALSE;
    }

    #if DEBUG
            cout<<"\n File creation : Success \n";
    #endif // DEBUG

    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );

    log_file<<"\n Restricted Boltzmann Machine \n"
            <<"\n Log Created on : "
            <<  now->tm_mday<<'-'
            << (now->tm_mon + 1) << '-'
            << (now->tm_year + 1900)<<" "
            << now->tm_hour<<':'
            << now->tm_min<<':'
            << now->tm_sec<<" Local Time"
            <<" \n\n Initializing Restricted Boltzmann Machine...Success\n";
    log_file.close();

    return TRUE;
}

inline bool RBM::Check_file(const string &name)
{
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


bool RBM::Get_netstat()
{
    return net_stat;
}

void RBM::Set_netstat(bool data)
{
    net_stat=data;

}

void RBM::Set_ready_to_train(bool data)
{
    ready_to_train = data;
}

bool RBM::Get_ready_to_train()
{
    return ready_to_train;
}

bool RBM::Init_RBM(uint16_t no_hidden, uint16_t no_visible, double alpha=0.1)
{
    if(no_hidden <=0 || no_visible <=0)
    {
       Set_netstat(FALSE);
       return FALSE;
     }
    else
    {
        num_hidden = no_hidden;
        num_visible = no_visible;
        learning_rate = alpha;
        Set_netstat(TRUE);

        #if FILE
            log_file.open("RBM_Log_File.txt",ios::app);
            log_file<<"\n Number of Hidden Neurons  : "<<num_hidden
                    <<"\n Number of Visible Neurons : "<<num_visible
                    <<"\n Learning Rate             : "<<learning_rate
                    <<"\n";
            log_file.close();
        #endif // FILE
    }

    return TRUE;
}

bool RBM::Init_weights(char *type)
{
    if(Get_netstat())
    {
        weights.resize(num_visible+1);

        for(uint16_t i=0; i<=num_visible;++i)
        {
            weights[i].resize(num_hidden+1);
            for(uint16_t j=0; j<=num_hidden;++j)
             {
                weights[i][j] = (j&&i)?standard_deviation*generate_random(0.0,1.0):0.0;

             }
        }
     }

    else
    {
         cout<<"\n Error: RBM haven't been initialized yet!! \n";

         return FALSE;
    }

    return TRUE;
}

bool RBM::Init_bias(char *type)
{
    if(Get_netstat())
    {
        if((!strcmp(type,"zeros")))
        {
            bias_init_type = 0;
        }

        else if((!strcmp(type,"random")))
        {
            bias_init_type = 1;
        }
        else
        {
            cout<<" Error: Invalid input argument for Init_bias()\n";

            return FALSE;
        }
    }
    else
    {
         cout<<"\n Error: RBM haven't been initialized yet!! \n";

         return FALSE;
    }

    return TRUE;
}

void RBM::Set_data_bias()
{
    vect_bool::iterator it;

    for(uint16_t i=0;i<train_data_rows;++i)
    {
        it = data[i].begin();
        data[i].insert(it,1);
    }

    train_data_cols = data[0].size();


}

inline void RBM::Config_activations()
{
    /* Configuring the Positive Hidden Activations */
    pos_hidden_activations.resize(train_data_rows);
    /* Configuring the Negative Hidden Activations */
    neg_hidden_activations.resize(train_data_rows);
    /* Configuring the Negative Visible Activations */
    neg_visible_activations.resize(train_data_rows);

    for(uint16_t i=0;i<train_data_rows;++i)
    {
        pos_hidden_activations[i].resize(num_hidden+1);
        neg_hidden_activations[i].resize(num_hidden+1);
        neg_visible_activations[i].resize(num_visible+1);
    }

   #if DEBUG

        cout<<"\n Pos_Hidden_Activation dimension : "<<pos_hidden_activations.size()
            <<" * "<<pos_hidden_activations[0].size()
            <<"\n Neg_Hidden_Activation dimension : "<<neg_hidden_activations.size()
            <<" * "<<neg_hidden_activations[0].size()
            <<"\n Neg_Visible_Activation dimension: "<<neg_visible_activations.size()
            <<" * "<<neg_visible_activations[0].size()<<"\n";
    #endif // DEBUG

    #if FILE
        log_file.open("RBM_Log_File.txt",ios::app);
        log_file<<"\n Pos_Hidden_Activation dimension : "<<pos_hidden_activations.size()
                <<" * "<<pos_hidden_activations[0].size()
                <<"\n Neg_Hidden_Activation dimension : "<<neg_hidden_activations.size()
                <<" * "<<neg_hidden_activations[0].size()
                <<"\n Neg_Visible_Activation dimension: "<<neg_visible_activations.size()
                <<" * "<<neg_visible_activations[0].size()<<"\n";
        log_file.close();
    #endif // FILE
}

inline void RBM::Config_probs()
{
    /* Configuring the Positive Hidden Probabilities */
    pos_hidden_probs.resize(train_data_rows);
    /* Configuring the Negative Hidden Probabilities */
    neg_hidden_probs.resize(train_data_rows);
    /* Configuring the Negative Visible Probabilities */
    neg_visible_probs.resize(train_data_rows);

    for(uint16_t i=0;i<train_data_rows;++i)
    {
        pos_hidden_probs[i].resize(num_hidden+1);
        neg_hidden_probs[i].resize(num_hidden+1);
        neg_visible_probs[i].resize(num_visible+1);
    }

   #if DEBUG

        cout<<"\n Pos_Hidden_Probs dimension : "<<pos_hidden_probs.size()
            <<" * "<<pos_hidden_probs[0].size()
            <<"\n Neg_Hidden_Probs dimension : "<<neg_hidden_probs.size()
            <<" * "<<neg_hidden_probs[0].size()
            <<"\n Neg_Visible_Probs dimension: "<<neg_visible_probs.size()
            <<" * "<<neg_visible_probs[0].size()<<"\n";
    #endif // DEBUG

    #if FILE
        log_file.open("RBM_Log_File.txt",ios::app);
        log_file<<"\n Pos_Hidden_Probs dimension : "<<pos_hidden_probs.size()
                <<" * "<<pos_hidden_probs[0].size()
                <<"\n Neg_Hidden_Probs dimension : "<<neg_hidden_probs.size()
                <<" * "<<neg_hidden_probs[0].size()
                <<"\n Neg_Visible_Probs dimension: "<<neg_visible_probs.size()
                <<" * "<<neg_visible_probs[0].size()<<"\n";
        log_file.close();
    #endif // FILE
}

inline void RBM::Config_associations()
{
    /* Configuring the Positive Associations */
    pos_associations.resize(num_visible+1);
    /* Configuring the Negative Associations */
    neg_associations.resize(num_visible+1);

    for(uint16_t i=0;i<num_visible+1;++i)
    {
        pos_associations[i].resize(num_hidden+1);
        neg_associations[i].resize(num_hidden+1);
    }

   #if DEBUG

        cout<<"\n Pos_Associations dimension : "<<pos_associations.size()
            <<" * "<<pos_associations[0].size()
            <<"\n Neg_Associations dimension : "<<neg_associations.size()
            <<" * "<<neg_associations[0].size()<<"\n";

    #endif // DEBUG

    #if FILE
        log_file.open("RBM_Log_File.txt",ios::app);
        log_file<<"\n Pos_Associations dimension : "<<pos_associations.size()
                <<" * "<<pos_associations[0].size()
                <<"\n Neg_Associations dimension : "<<neg_associations.size()
                <<" * "<<neg_associations[0].size()<<"\n";
        log_file.close();
    #endif // FILE
}

inline void RBM::Config_hiddden_states()
{
    /* Configuring the Positive Hidden Activations */
    pos_hidden_states.resize(train_data_rows);

    for(uint16_t i=0;i<train_data_rows;++i)
        pos_hidden_states[i].resize(num_hidden+1);

    #if DEBUG

        cout<<"\n Pos_Hidden_States dimension : "<<pos_hidden_states.size()
            <<" * "<<pos_hidden_states[0].size()<<"\n";
    #endif // DEBUG

    #if FILE
        log_file.open("RBM_Log_File.txt",ios::app);
        log_file<<"\n Pos_Hidden_States dimension : "<<pos_hidden_states.size()
                <<" * "<<pos_hidden_states[0].size()<<"\n";
        log_file.close();
    #endif // FILE
}

inline void RBM::Compute_pos_hidden_activations()
{
    #if DEBUG
        cout<<"\n Data + bias dimensions: "<<data.size()<<" * "
            <<data[0].size()
            <<"\n Weight dimensions: "<<weights.size()<<" * "
            <<weights[0].size()
            <<"\n";
    #endif // DEBUG

    double sum=0;
    /* Data * Weights */
    for(uint16_t x=0;x<train_data_rows;++x)
    {
        for(uint16_t y=0; y<(num_hidden+1);++y)
        {
            sum=0.0;
            for(uint16_t z=0; z< train_data_cols;++z)
                sum += (double)data[x][z] * weights[z][y];
            pos_hidden_activations[x][y]=sum;

         }
     }
}

inline void RBM::Compute_neg_hidden_activations()
{
    #if DEBUG
        cout<<"\n Neg_Visible_Probs dimensions: "<<neg_visible_probs.size()
            <<" * "<<neg_visible_probs[0].size()
            <<"\n Weight dimensions: "<<weights.size()<<" * "
            <<weights[0].size()
            <<"\n";
    #endif // DEBUG

    double sum=0;
    /* Negative Visible Probabilities * Weights */
    for(uint16_t x=0;x<train_data_rows;++x)
    {
        for(uint16_t y=0; y<(num_hidden+1);++y)
        {
            sum=0.0;
            for(uint16_t z=0; z< train_data_cols;++z)
                sum += neg_visible_probs[x][z] * weights[z][y];
            neg_hidden_activations[x][y]=sum;

         }
     }
}

inline void RBM::Compute_pos_associations()
{
     double sum=0;
     /* Transpose(data) * Positive Hidden Probabilities - Row-wise */
     for(uint16_t x=0;x<data[0].size();++x)
     {
         for(uint16_t y=0; y<pos_hidden_probs[0].size();++y)
         {
             sum=0.0;
             for(uint16_t z=0; z< data.size();++z)
                 sum += (double)data[z][x] * pos_hidden_probs[z][y];
             pos_associations[x][y]=sum;

          }
     }

}

inline void RBM::Compute_neg_associations()
{
     #if DEBUG
        cout<<"\n Transpose(Neg_visible_Probs) dimensions: "<<neg_visible_probs[0].size()
            <<" * "<<neg_visible_probs.size()
            <<"\n Pos_hidden_Probs dimensions: "<<neg_hidden_probs.size()
            <<" * "<<neg_hidden_probs[0].size()
            <<"\n";
    #endif // DEBUG

     double sum=0;
     /* Transpose(Negative Visible Probabilities) * Negative Hidden Probabilities - Row-wise */
     for(uint16_t x=0;x<neg_visible_probs[0].size();++x)
     {
         for(uint16_t y=0; y<neg_hidden_probs[0].size();++y)
         {
             sum=0.0;
             for(uint16_t z=0; z< neg_visible_probs.size();++z)
                 sum += neg_visible_probs[z][x] * neg_hidden_probs[z][y];
             neg_associations[x][y]=sum;

          }
     }

}

inline void RBM::Compute_neg_visible_activations()
{
    #if DEBUG
        cout<<"\n Pos_hidden_States dimensions: "<<pos_hidden_states.size()
            <<" * "<<pos_hidden_states[0].size()
            <<"\n Transpose(Weights) dimensions: "<<weights[0].size()
            <<" * "<<weights.size()
            <<"\n";
    #endif // DEBUG

    double sum=0;
    /* Positive hidden states * Transpose(Weights)- column-wise */
    for(uint16_t x=0;x<train_data_rows;++x)
    {
        for(uint16_t y=0; y<(num_visible+1);++y)
        {
            sum=0.0;
            for(uint16_t z=0; z< num_hidden+1;++z)
                sum += pos_hidden_states[x][z] * weights[y][z];
            neg_visible_activations[x][y]=sum;
         }
     }

}

void RBM::Compute_pos_hidden_states()
{
    for(uint16_t i=0;i<pos_hidden_states.size();++i)
    {
        for(uint16_t j=0;j<pos_hidden_states[i].size();++j)
        {
            pos_hidden_states[i][j] = (pos_hidden_probs[i][j] > generate_random(0.0,1.0))? TRUE:FALSE;
        }
    }
}

/*void RBM::Compute_pos_visible_states()
{
	for (uint16_t i = 0;i<data.size();++i)
	{
		data[0][i] = TRUE;
		for (uint16_t j = 1;j<data[0].size();++j)
		{
			data[i][j] = (pos_visible_probs[i][j] > generate_random(0.0, 1.0)) ? TRUE : FALSE;
		}
	}
}*/

void RBM::Set_neg_visible_probs_bias()
{
	for (uint16_t i = 0;i < neg_visible_probs.size();++i)
		neg_visible_probs[i][0] = 1;
}

bool RBM::RBM_train(uint32_t epchs, bool method)
{
    epochs = epchs;
    uint16_t nrows = data.size();
    uint16_t ncols = data[0].size();

    train_data_rows = nrows;
    train_data_cols = ncols;

    #if DEBUG
        cout<<"\n RBM Data dimensions  : "<<train_data_rows<<" * "<<train_data_cols
            <<"\n Weight dimensions: "<<num_visible+1<<" * "<<num_hidden+1<<"\n"
            <<"\n Initial RBM Data \n";
    #endif //DEBUG

    #if FILE
        log_file.open("RBM_Log_File.txt",ios::app);
        log_file<<"\n RBM Data Dimensions:"<<train_data_rows
                <<" * "<<train_data_cols
                <<"\n Weight dimensions: "<<num_visible+1<<" * "<<num_hidden+1<<"\n"
                <<"\n Initial Data \n";
        log_file.close();
    #endif // FILE

    Display_data();

    if(ncols!= (num_visible))
    {
        cout<<" Error: Invalid data dimensions\n";

        return FALSE;
    }

    else
    {
        if(Get_netstat())
        {
            /** Set Biases for the data as 1 **/
            Set_data_bias();

            #if DEBUG
                cout<<"\n RBM Data dimensions with bias: "<<train_data_rows
                    <<" * "<<train_data_cols<<"\n";
            #endif //DEBUG

            #if FILE
              log_file.open("RBM_Log_File.txt",ios::app);
              log_file<<"\n RBM Data + Bias Dimensions: "<<train_data_rows
                      <<" * "<<train_data_cols<<"\n"
                      <<"\n RBM Data with Biases \n";
              log_file.close();
            #endif // FILE

            /** Display Data **/
            cout<<"\n RBM Data with Biases\n";

            Display_data();

            Display_weights();

            /** Configure RBM Parameters **/
            Config_activations();
            Config_probs();
            Config_associations();
            Config_hiddden_states();
            Set_ready_to_train(TRUE);
            Config_error();
			
            if(Get_ready_to_train())
            {

				/** Train Data **/
				#if DEBUG
					cout << "\n Training RBM ...\n\n";
				#endif // DEBUG

				#if FILE
					log_file.open("RBM_Log_File.txt", ios::app);
					log_file << "\n Training RBM ...\n\n";
					log_file.close();
				#endif // FILE

				time_t start_time = time(0);   // get time now

				for(curr_epoch=0;curr_epoch<epochs;++curr_epoch)
                {

                    #if DEBUG
                        cout<<"\n Epoch : "<<curr_epoch+1
                            <<"\n";
                    #endif // DEBUG

                    #if FILE
                        log_file.open("RBM_Log_File.txt",ios::app);
                        log_file<< "\n Epoch : "<<curr_epoch+1
                                <<"\n";
                        log_file.close();
                    #endif // FILE

					

					/* Gibbs Sampling */
					for (uint16_t k = 0;k < 15;++k)
					{

						// Data is simply the positive visible state

						Compute_pos_hidden_activations();
						//Display_Pos_hidden_activation();

						Compute_probs(1);
						//Display_Pos_hidden_probs();

						Compute_pos_hidden_states();
						//Display_Pos_hidden_States();

						/*Compute_pos_associations();
						Display_Pos_associations();*/

						/* Reconstruction of the visible unit from the hidden units*/
						Set_neg_visible_probs_bias();
						//Display_Neg_visible_probs(2);

						Compute_neg_visible_activations();
						//Display_Neg_visible_activation(2);

						Compute_probs(3);
						//Display_Neg_visible_probs(2);

						Compute_neg_hidden_activations();
						//Display_Neg_hidden_activation();

						Compute_probs(2);
						//Display_Neg_hidden_probs();

						/*Compute_neg_associations();
						Display_Neg_associations();

						Update_weights();

						Update_error();*/
					}

					if (curr_epoch >= 0.75*epochs)
						learning_rate = 0.32;
					else if (curr_epoch >= 0.95*epochs)
						learning_rate = 0.42;

					Compute_pos_associations();
					//Display_Pos_associations();

					Compute_neg_associations();
					//Display_Neg_associations();

					Update_weights();

					Update_error();
                }
				
				time_t end_time = time(0) - start_time;   // get time now
				struct tm * now = localtime(&end_time);

				#if DEBUG
					cout<< "\n Training Complete \n"
						<< "\n Elpased Time : "
						<< now->tm_hour << ':'
						<< now->tm_min << ':'
						<< now->tm_sec;						
				#endif //DEBUG

				#if FILE 
					log_file.open("RBM_Log_File.txt", ios::app);
					log_file<< "\n Training Complete \n"
							<< "\n Elpased Time : "
							<< now->tm_hour << ':'
							<< now->tm_min << ':'
							<< now->tm_sec;
					log_file.close();
				#endif //FILE

				Display_error(5,"fixed");
								
            }

       }

        else
        {
            cout<<"\n Error: RBM haven't been initialized yet!! \n";

            return FALSE;
        }
    }
    return TRUE;
}

void RBM::Update_weights()
{
    for(uint16_t i=0;i<weights.size();++i)
    {
        for(uint16_t j=0;j<weights[0].size();++j)
            weights[i][j]+=learning_rate*(pos_associations[i][j]-neg_associations[i][j]);
    }
}

void RBM::Update_error()
{
    error[curr_epoch]=0.0;
    for(uint16_t i=0;i<neg_visible_probs.size();++i)
    {
        for(uint16_t j=0;j<neg_visible_probs[0].size();++j)
            error[curr_epoch] += pow(((double)data[i][j]- neg_visible_probs[i][j]),2);
    }

}

double RBM::Logistic(double value)
{
    return(1.0/(1.0+exp(-1.0*value)));
}

void RBM::Compute_probs(uint8_t flag)
{
    switch(flag)
    {
        case 1:
            /* Calculate the probabilities for positive hidden activations */
            for(uint16_t i=0; i<pos_hidden_activations.size();++i)
            {
                for(uint16_t j=0;j<pos_hidden_activations[0].size();++j)
                {
                    pos_hidden_probs[i][j]=Logistic(pos_hidden_activations[i][j]);
                }
            }
            break;
        case 2:
            /* Calculate the probabilities for negative hidden activations */
            for(uint16_t i=0; i<neg_hidden_activations.size();++i)
            {
                for(uint16_t j=0;j<neg_hidden_activations[0].size();++j)
                {
                    neg_hidden_probs[i][j]=Logistic(neg_hidden_activations[i][j]);
                }
            }
            break;
        case 3:
            /* Calculate the probabilities for negative visible activations */
            for(uint16_t i=0; i<neg_visible_activations.size();++i)
            {
                for(uint16_t j=0;j<neg_visible_activations[0].size();++j)
                {
                    neg_visible_probs[i][j]=Logistic(neg_visible_activations[i][j]);
                }
            }
            break;
        default:
            cout<<"\n Error: Invalid flag\n";

    }
}

void RBM::Display_Pos_hidden_activation(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Positive Hidden Activations\n";
        for(uint16_t i=0;i<pos_hidden_activations.size();++i)
        {
            for(uint16_t j=0;j<pos_hidden_activations[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<pos_hidden_activations[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Pos_hidden_activation()\n";
}

void RBM::Display_Neg_hidden_activation(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Negative Hidden Activations\n";
        for(uint16_t i=0;i<neg_hidden_activations.size();++i)
        {
            for(uint16_t j=0;j<neg_hidden_activations[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<neg_hidden_activations[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Neg_hidden_activation()\n";
}

void RBM::Display_Neg_visible_activation(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Negative Visible Activations\n";
        for(uint16_t i=0;i<neg_visible_activations.size();++i)
        {
            for(uint16_t j=0;j<neg_visible_activations[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<neg_visible_activations[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Neg_visible_activation()\n";
}

void RBM::Display_Neg_hidden_probs(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Negative Hidden Probabilities\n";
        for(uint16_t i=0;i<neg_hidden_probs.size();++i)
        {
            for(uint16_t j=0;j<neg_hidden_probs[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<neg_hidden_probs[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Neg_hidden_probs()\n";
}

void RBM::Display_Neg_visible_probs(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Negative Visible Probabilities\n";
        for(uint16_t i=0;i<neg_visible_probs.size();++i)
        {
            for(uint16_t j=0;j<neg_visible_probs[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<neg_visible_probs[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Neg_hidden_probs()\n";
}

void RBM::Display_Pos_hidden_probs(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Positive Hidden Probabilities\n";
        for(uint16_t i=0;i<pos_hidden_probs.size();++i)
        {
            for(uint16_t j=0;j<pos_hidden_probs[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<pos_hidden_probs[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Pos_hidden_probs()\n";
}

void RBM::Display_Pos_hidden_States(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Positive Hidden States\n";
        for(uint16_t i=0;i<pos_hidden_states.size();++i)
        {
            for(uint16_t j=0;j<pos_hidden_states[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<pos_hidden_states[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Pos_hidden_States()\n";
}

void RBM::Display_Pos_associations(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Positive Associations \n";
        for(uint16_t i=0;i<pos_associations.size();++i)
        {
            for(uint16_t j=0;j<pos_associations[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<pos_associations[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Pos_associations()\n";
}

void RBM::Display_Neg_associations(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {
        cout<<"\n Negative Associations \n";
        for(uint16_t i=0;i<neg_associations.size();++i)
        {
            for(uint16_t j=0;j<neg_associations[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<neg_associations[i][j]<<"  ";
            }
            cout<<"\n";
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Neg_associations()\n";
}

void RBM::Display_data(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {

        for(uint16_t i=0;i<data.size();++i)
        {
            for(uint16_t j=0;j<data[0].size();++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<data[i][j]<<"  ";

                #if FILE
                    log_file.open("RBM_Log_File.txt",ios::app);
                    (!strcmp(notation,"fixed"))? log_file<<std::fixed :
                                             log_file<<std::scientific;
                    log_file<<setprecision(precision)<<data[i][j]<<"  ";
                    log_file.close();

                #endif // FILE
            }
            cout<<"\n";
            #if FILE
                    log_file.open("RBM_Log_File.txt",ios::app);
                    log_file<<"\n";
                    log_file.close();

            #endif // FILE
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Weights()\n";
}

void RBM::Display_weights(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {

        cout<<"\n Weights & Biases \n";

        #if FILE
              log_file.open("RBM_Log_File.txt",ios::app);
              log_file<<"\n Weights & Biases \n";
              log_file.close();
        #endif // FILE

        for(uint16_t i=0;i<=num_visible;++i)
        {
            for(uint16_t j=0;j<=num_hidden;++j)
            {
                (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
                cout<<setprecision(precision)<<weights[i][j]<<"  ";

                #if FILE
                    log_file.open("RBM_Log_File.txt",ios::app);
                    (!strcmp(notation,"fixed"))? log_file<<std::fixed :
                                             log_file<<std::scientific;
                    log_file<<setprecision(precision)<<weights[i][j]<<"  ";
                    log_file.close();

                #endif // FILE
            }
            cout<<"\n";
            #if FILE
                    log_file.open("RBM_Log_File.txt",ios::app);
                    log_file<<"\n";
                    log_file.close();

            #endif // FILE
        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Weights()\n";
}

void RBM::Display_error(uint8_t precision, char *notation)
{
    if(precision>0 && ((!strcmp(notation,"fixed"))||(!strcmp(notation,"scientific"))))
    {

        cout<<"\n Error \n";

        #if FILE
              log_file.open("RBM_Log_File.txt",ios::app);
              log_file<<"\n Error \n";
              log_file.close();
        #endif // FILE

        for(uint16_t i=0;i<epochs;++i)
        {
            (!strcmp(notation,"fixed"))? cout<<std::fixed :
                                             cout<<std::scientific;
            cout<<setprecision(precision)<<error[i]<<"\n";

            #if FILE
                log_file.open("RBM_Log_File.txt",ios::app);
                (!strcmp(notation,"fixed"))? log_file<<std::fixed :
                                         log_file<<std::scientific;
                log_file<<setprecision(precision)<<error[i]<<"\n";
                log_file.close();
            #endif // FILE

        }
    }
    else
        cout<<"\n Error: Invalid input arguments for Display_Weights()\n";
}

bool RBM::Get_data(matrix_bool &arr, uint16_t nrows)
{
    data.resize(nrows);

    #if DEBUG
        cout<<"\n Input data dimensions: "<<nrows<<" * "<<arr[0].size();
    #endif // DEBUG

    #if FILE
              log_file.open("RBM_Log_File.txt",ios::app);
              log_file<<"\n Input Data Dimensions: "
                      <<nrows<<" * "<<arr[0].size();;
              log_file.close();
    #endif // FILE

    if(arr[0].size()!= num_visible)
    {
        cout<<"\n Error: Invalid input arguments for Get_data()\n";
        #if FILE
              log_file.open("RBM_Log_File.txt",ios::app);
              log_file<<"\n Error: Invalid input arguments for Get_data()\n";
              log_file.close();
        #endif // FILE

        return FALSE;
    }

    for(int i=0;i<nrows;++i)
    {
        data[i].resize(num_visible);
        for(int j=0;j<num_visible;++j)
            data[i][j]=arr[i][j];
    }

    return TRUE;
}

int main()
{
    RBM bolt_net;
    uint16_t hidden = 2, visible =6, nrows = 6;

    bolt_net.Init_RBM(hidden,visible);
	bolt_net.set_std(0.1);
    bolt_net.Init_weights();

    bool arr[6][6]={{1,1,1,0,0,0},
                  {1,0,1,0,0,0},
                  {1,1,1,0,0,0},
                  {0,0,1,1,1,0},
                  {0,0,1,1,0,0},
                  {0,0,1,1,1,0}};

    matrix_bool data;
    data.resize(nrows);

    for(int i=0;i<nrows;++i)
    {
        data[i].resize(visible);
        for(int j=0;j<visible;++j)
            data[i][j]=arr[i][j];
    }

    bolt_net.Get_data(data,nrows);
    bolt_net.RBM_train(10);
	
    return 0;
}