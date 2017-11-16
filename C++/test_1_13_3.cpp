#include <iostream>
using namespace std;
int main(){
    int begin, end;
    cout<<"please input 2 integers."<<"\n";
    cin>>begin>>end;
    if(end < begin){
        int tmp = begin;
        begin = end;
        end = tmp;
    }
    for(int i = begin; i <= end; ++i){
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
