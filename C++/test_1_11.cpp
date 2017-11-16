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
    while(begin<=end){
        cout<<begin<<"\n";
        begin++;
    }
    return 0;
}
