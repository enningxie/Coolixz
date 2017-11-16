#include <iostream>
using namespace std;
int main()
{
    int sum = 0;
    int i = 50;
    while(i<101){
        sum += i;
        ++i;
    }
    cout<<sum<<endl;
    return 0;
}
