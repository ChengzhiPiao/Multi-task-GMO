#pragma GCC optimize(2)
#include<bits/stdc++.h>
#define N 25
#define K 1000
#define rand_time 20
#define LL long long
#define inf 21474836477777
#define INF 2147483647214748364
#define LD long double
#define eps 1e-4
#define max_dif 1000000
#define dif 1000
#define s_id set<int>::iterator
using namespace std;
int n,m,f[N],rk[N],v[N];bool vis[N];
struct graph{int id,n,m,u[N*N],v[N*N];string labels[N];}GA,GB,GR[705];
inline int read(){char ch=getchar();while(!isdigit(ch)) ch=getchar();int res=0;while(isdigit(ch)) res=res*10+ch-'0',ch=getchar();return res;}
inline bool cmp(int x,int y){return v[x]<v[y];}
int main(){
    // ios::sync_with_stdio(false);
    // cin.tie(0); cout.tie(0);
    // freopen("data_used.txt","r",stdin);
    // for(int i=1,x;i<=100;i++) cin>>x,use[x]=1;
    freopen("graph.txt","r",stdin);
    for(int i=0;i<700;i++){
        // cout<<i<<'\n';
        GR[i].id=read(),GR[i].n=read();
    }
    freopen("ground_truth_ori.txt","r",stdin);
    freopen("ground_truth.txt","w",stdout);
    
    clock_t start_time=clock();

    srand(time(NULL));
    int tc=1;
    // cin>>tc;
    tc=700*700;cout<<tc<<'\n';
    for(int i=0;i<tc;i++){
        GA=GR[i/700],GB=GR[i%700],n=GA.n,m=GB.n;cout<<GA.id<<' '<<GB.id<<'\n';
        int mcs,tot;mcs=read(),tot=read();cout<<mcs<<' '<<tot<<'\n';
        while(tot--){
            for(int j=0;j<n;j++) f[j]=-1,v[j]=rand(),rk[j]=j;for(int j=0;j<m;j++) vis[j]=0;
            sort(rk,rk+n,cmp);
            char ch=getchar();while(ch!='\n'){
                while(!isdigit(ch)){ch=getchar();if(ch=='\n') break;}if(ch=='\n') break;int x=0;
                while(isdigit(ch)) x=x*10+ch-'0',ch=getchar();
                while(!isdigit(ch)){ch=getchar();if(ch=='\n') break;}if(ch=='\n') break;int y=0;
                while(isdigit(ch)) y=y*10+ch-'0',ch=getchar();//cout<<(int)ch<<' '<<x<<' '<<y<<' ';
                f[x]=y,vis[y]=1;
            }
            // cout<<'\n';
            // for(int j=1,x,y;j<=tot;j++) cin>>x>>y,f[x]=y,vis[y]=1;
            int k=0;for(int j=0;j<n;j++) if(f[j]==-1) f[j]=m,k++;
            // int k=0;for(int j=0;j<n;j++) if(f[j]==-1) f[j]=m,k++;
            for(int j=0;j<n;j++) cout<<f[j]<<' ';cout<<'\n';
        }
    }

    // freopen("set_random_matching.txt","w",stdout);
    
    // for(int i=1;i<=K/100;i++) printf("%.3f ",1.0*bm[i]/10000.0);

    // for(int i=1;i<=K/100;i++) printf("%.2f ",ged[i]/100.0);

    clock_t end_time=clock();

    // printf("%.4f\n",(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC)/tc);
    return 0;
}
