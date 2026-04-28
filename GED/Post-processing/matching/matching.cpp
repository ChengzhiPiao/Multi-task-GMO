#pragma GCC optimize(2)
#include<bits/stdc++.h>
#define N 305
#define K 300000
#define K_max 100
#define rand_time 3
#define LL long long
#define inf 2147483647777777
#define INF 2147483647214748364
#define LD long double
#define eps 1e-7
#define s_id set<int>::iterator
using namespace std;
int n,m,mapping[N][N],rk[K+5];LL cost[N][N],c[N][N];// Cost Matrix
bool con1[N][N],con2[N][N];
LL dis[N<<1],slack[N];int vis[N<<1],vis2[N<<1];bool in[N];
int tot,tt,vx[N],lst[N],ok[N][N],hd,tl,q[N*N*N];// for choosing a random vertex
bool use[K+5];
int ged[N];LL bm[K_max+5];
struct graph{int n,m,u[N*N],v[N*N];string labels[N];}GA,GB;
struct Graph{
    vector<pair<int,int> > I,O;// Matching M must contain I, and must not contain O
    int best_matching[N],second_matching[N],match[N],V;LL bm,sm;
    LL h[N<<1];// result of KM algorithm, for setting the edges' weight non-negative
    inline void print(){
        // cout/*<<"matching result: "*/<<bm<<' '/*<<"matching matrix:\n"*/;
        // for(int i=1;i<=n;cout<<'\n',i++) for(int j=1;j<=m;j++) if(best_matching[i]==j) cout<<1<<' ';else cout<<0<<' ';cout<<'\n';
    } // print the best matching
    inline void spfa(){
        int S=0;for(int i=1;i<=n;i++) h[i]=-inf,(!vis[i])&&(S=i,0);if(!S) return;h[S]=0;hd=1,tl=1;q[1]=S;
        while(hd<=tl){
            int x=q[hd++],y=best_matching[x];in[x]=0;
            for(int i=1;i<=n;i++) if(!vis[i]&&ok[i][y]!=tt&&h[i]<h[x]+cost[i][y]-cost[x][y])
            {h[i]=h[x]+cost[i][y]-cost[x][y];if(!in[i]) in[i]=1,q[++tl]=i;}
        }
    }
    inline void get_h(){// get the dual variable h
        tt++;for(int i=0;i<O.size();i++) ok[O[i].first][O[i].second]=tt;spfa();
        for(int i=1;i<=n;i++) h[best_matching[i]+n]=cost[i][best_matching[i]]-h[i];
    }
    inline void dijkstra(int S){
        for(int i=1;i<=n+m;i++) dis[i]=INF;dis[S]=0,vis[S]=1;while(1){
            if(S<=n){for(int i=1;i<=m;i++) if(i!=best_matching[S]&&dis[S]+c[S][i]<dis[i+n]&&!vis[i+n]&&c[S][i]<inf) dis[i+n]=dis[S]+c[S][i],lst[i]=S;}
            else{if(match[S-n]) dis[match[S-n]]=dis[S];} 
            S=0;for(int i=1;i<=n+m;i++) if(!vis[i]&&dis[i]<INF) if(!S||dis[i]<dis[S]) S=i;if(!S) break;vis[S]=1;
        }
    }
    inline void get_second(){
        memset(second_matching,0,sizeof(second_matching)),memset(match,0,sizeof(match)),memset(vis,0,sizeof(vis));
        for(int i=0;i<I.size();i++) vis[I[i].first]=vis[I[i].second+n]=2;
        get_h();for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) if(best_matching[i]==j) c[i][j]=0,match[j]=i;else c[i][j]=h[i]+h[j+n]-cost[i][j];
        for(int i=0;i<O.size();i++) c[O[i].first][O[i].second]=inf;
        tot=0;for(int i=1;i<=n;i++) if(!vis[i]) vx[++tot]=i;if(!tot){sm=-INF;return;} memcpy(vis2,vis,sizeof(vis));
        V=0,sm=-INF;for(int i=1;i<=tot;i++){
            memcpy(vis,vis2,sizeof(vis2));
            int S=vx[i],x=best_matching[S];dijkstra(S);if(dis[x+n]<INF&&bm-dis[x+n]>sm){
                sm=bm-dis[x+n],V=S,memcpy(second_matching,best_matching,sizeof(best_matching));
                while(1){second_matching[lst[x]]=x;if(lst[x]==S) break;x=best_matching[lst[x]];} // find the cycle and get the second-best-matching
            }
        }
    }
    inline bool check(int x){
        vis[x+n]=1;if(match[x]!=-1){q[++tl]=match[x],vis[match[x]]=1;return 0;}
        while(x!=-1) match[x]=lst[x],swap(x,best_matching[lst[x]]);return 1;
    }
    inline void bfs(int S){
        hd=1,tl=1;q[1]=S,vis[S]=1;while(1){
            while(hd<=tl){
                int x=q[hd++];for(int i=1;i<=n;i++) if(!vis[i+n]){
                    LL d=h[x]+h[i+n]-cost[x][i];if(slack[i]>=d)
                    {lst[i]=x;if(d) slack[i]=d;else if(check(i)) return;}
                }
            }
            LL res=INF;for(int i=1;i<=n;i++) if(!vis[i+n]) res=min(res,slack[i]);
            for(int i=1;i<=n;i++){if(vis[i]) h[i]-=res;if(vis[i+n]) h[i+n]+=res;else slack[i]-=res;}
            for(int i=1;i<=n;i++) if(!vis[i+n]&&!slack[i]&&check(i)) return;
        }
    }
    inline void KM(){
        for(int i=1;i<=n;i++){match[i]=best_matching[i]=-1,h[i]=h[i+n]=0;for(int j=1;j<=n;j++) h[i]=max(h[i],cost[i][j]);}
        for(int i=1;i<=n;i++){for(int j=1;j<=n;j++) slack[j]=inf;memset(vis,0,sizeof(vis)),bfs(i);}
        bm=0;for(int i=1;i<=n;i++) bm+=cost[i][best_matching[i]];
    }
}G[K_max+5];// K subgraphs
inline bool cmp(int x,int y){return G[x].bm>G[y].bm;}
inline void remove_edge(){
    int mm=0;for(int i=1;i<=GA.m;i++) if(!con1[GA.u[i]][GA.v[i]]) con1[GA.v[i]][GA.u[i]]=con1[GA.u[i]][GA.v[i]]=1,mm++,GA.u[mm]=GA.u[i],GA.v[mm]=GA.v[i];GA.m=mm;
    mm=0;for(int i=1;i<=GB.m;i++) if(!con2[GB.u[i]][GB.v[i]]) con2[GB.v[i]][GB.u[i]]=con2[GB.u[i]][GB.v[i]]=1,mm++,GB.u[mm]=GB.u[i],GB.v[mm]=GB.v[i];GB.m=mm;
    for(int i=1;i<=GA.n;i++) for(int j=1;j<=GA.n;j++) con1[i][j]=0;for(int i=1;i<=GB.n;i++) for(int j=1;j<=GB.n;j++) con2[i][j]=0;
}
inline void read(){
    cin>>n>>m;for(int i=1;i<=n;i++){double x;for(int j=1;j<=m;j++) cin>>x,cost[i][j]=(LL)x;}
    for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) cin>>mapping[i][j];
    GA.n=n;for(int i=1;i<=n;i++) cin>>GA.labels[i];cin>>GA.m;for(int i=1;i<=GA.m;i++) cin>>GA.u[i]>>GA.v[i],GA.u[i]++,GA.v[i]++;
    GB.n=m;for(int i=1;i<=m;i++) cin>>GB.labels[i];cin>>GB.m;for(int i=1;i<=GB.m;i++) cin>>GB.u[i]>>GB.v[i],GB.u[i]++,GB.v[i]++;
    // remove_edge();  note: this step is for IMDB dataset, where an undirected edge (u,v) appears twice as (u,v),(v,u) in the input graph
}
inline void init(){
    memset(vis,0,sizeof(vis));G[1].I.clear(),G[1].O.clear();
    for(int i=n+1;i<=m;i++) for(int j=1;j<=m;j++) cost[i][j]=0;n=m;
    // for(int i=1;i<=GA.n;i++) for(int j=1;j<=GB.n;j++) if(GA.labels[i]!=GB.labels[j]) cost[i][j]=0;
    G[1].KM();// get the best matching of the initial graph
    // for(int i=1;i<=n;cout<<'\n',i++) for(int j=1;j<=n;j++) cout<<cost[i][j]<<' ';
}
// inline void random_shuffle(){for(int i=1;i<=n;i++) swap(rk[rand()%n+1],rk[rand()%n+1]);}
int KK;double running_time[K+5][15];
inline void AKM(int tc){
    clock_t start_time=clock();
    bool flg=1;G[1].print();G[1].get_second();// Get the second-best matching
    for(int i=0;i<=10;i++) running_time[tc][i]=0;
    clock_t end_time=clock();
    running_time[tc][0]=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
    for(int k=2;k<=K_max;k++){
        int id=1;for(int i=2;i<k;i++) if(G[i].sm>G[id].sm) id=i;// find the maximum second_best_matching
        if(G[id].sm==-INF){KK=k-1;/*cout<<"Only "<<k-1<<" matchings exist"<<'\n'<<'\n';*/flg=0;for(int i=k;i<=K_max;i++) G[i].bm=-INF;break;}
        /*
        int v=0;for(int i=1;i<=n;i++) if(G[id].best_matching[i]^G[id].second_matching[i]) v=i;// choose an edge from{best_matching-second_matching}
        */
        G[k].I=G[id].I,G[id].I.push_back(make_pair(G[id].V,G[id].best_matching[G[id].V]));
        G[k].O=G[id].O,G[k].O.push_back(make_pair(G[id].V,G[id].best_matching[G[id].V]));
        memcpy(G[k].best_matching,G[id].second_matching,sizeof(G[id].second_matching)),G[k].bm=G[id].sm,G[k].print();
        G[id].get_second(),G[k].get_second();// update I&O and second_best_matching
        if(k%10==0){
            end_time=clock();
            running_time[tc][k/10]=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
        }
    }
    end_time=clock();
    for(int i=1;i<=10;i++){
        if(i*10>KK) running_time[tc][i]=max(running_time[tc][i],(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC));
    }
    // if(flg) cout<<K<<"-best matchings are found"<<'\n'<<'\n';
}
int MCS[K+5],gr[K+5],num,acc;double E,recall,prec,f1,precision;LL e;
struct path{
    bool del[N][N],add[N][N],out[N];int mapping[N];
    int gen_path(){// If the input graph has multiple edges, e.g. (u,v) appears more than once, then the way of calculating GED should be modified.
        memset(del,0,sizeof(del)),memset(add,0,sizeof(add)),memset(out,0,sizeof(out));for(int j=GA.n+1;j<=GB.n;j++) out[mapping[j]]=1;
        int res=GB.n-GA.n;for(int j=1;j<=GA.n;j++) if(GA.labels[j]!=GB.labels[mapping[j]]) res++;
        for(int j=1;j<=GB.m;j++) if(out[GB.u[j]]||out[GB.v[j]]) res++,del[GB.u[j]][GB.v[j]]=1;
        for(int j=1;j<=GA.n;j++) for(int k=j+1;k<=GA.n;k++) if(con1[j][k]^con2[mapping[j]][mapping[k]]) res++,del[j][k]=con1[j][k],add[mapping[j]][mapping[k]]=con2[mapping[j]][mapping[k]];
        return res;
    }
}gt,pred[K_max+5];
int pre_mapping[N];
inline void solve(int tc){
    KK=K_max;
    // cout<<"Test data "<<tc<<":\n"/*<<"ground-truth matching result: "*/;
    read();// read the data

    clock_t start_time=clock();

    // if(!use[tc]) return;
    init();// Get the matching matrix and the best matching
    AKM(tc);// Algorithm of K-best Matching
    // for(int i=1;i<=K;i++) gr[i]=G[i].bm;
    // for(int i=1;i<=K;i++) bm[(i-1)/100+1]+=G[i].bm,cout<<G[i].bm<<' ';cout<<'\n';
    for(int j=1;j<=GA.m;j++) con1[GA.u[j]][GA.v[j]]=con1[GA.v[j]][GA.u[j]]=1;for(int j=1;j<=GB.m;j++) con2[GB.u[j]][GB.v[j]]=con2[GB.v[j]][GB.u[j]]=1;
    // for(int j=1;j<=GA.n;j++) gt.mapping[j]=mapping[j];gt.gen_path();
    int ans=888888;bool flg=0;double recall_=0,prec_=0,f1_=0,precision_=0;
    for(int i=1;i<=KK;i++){
        for(int j=1;j<=n;j++) pred[i].mapping[j]=G[i].best_matching[j];int tmp=pred[i].gen_path();ans=min(ans,tmp);
        int res=0;for(int j=1;j<=GA.n;j++) if(mapping[j][G[i].best_matching[j]]) res++;
        if(ans==tmp) precision_=1.0*res/GA.n;

        if(ans==tmp){for(int j=1;j<=n;j++) pre_mapping[j]=G[i].best_matching[j];}

        // int intersec=0;
        // for(int j=1;j<=GA.n;j++) if(GA.labels[j]!=GB.labels[gt.mapping[j]]&&pred[i].mapping[j]==gt.mapping[j]) intersec++;
        // for(int j=1;j<=GB.n;j++) if(pred[i].out[j]&&gt.out[j]) intersec++;
        // for(int j=1;j<=GA.n;j++) for(int k=j+1;k<=GA.n;k++) if(pred[i].del[j][k]&&gt.del[j][k]) intersec++;
        // for(int j=1;j<=GB.n;j++) for(int k=j+1;k<=GB.n;k++) if(pred[i].add[j][k]&&gt.add[j][k]) intersec++;
        // if(gr[tc]){recall_=max(recall_,1.0*intersec/gr[tc]),prec_=max(prec_,1.0*intersec/tmp);if(intersec) f1_=max(f1_,2.0/(1.0*gr[tc]/intersec+1.0*tmp/intersec));}
        // if(ans==gr[tc]){flg=1,cout<<i<<'\n';break;}
        // if(i==1||i%10==0) cout<<ans<<' ';
    }

    cout<<"GED = "<<ans<<'\n';
    cout<<"Matching = ";for(int i=1;i<=GA.n;i++) if(pre_mapping[i]<=GB.n) cout<<"("<<i-1<<","<<pre_mapping[i]-1<<") ";cout<<'\n';
    // for(int i=1;i<=n;i++) cout<<pre_mapping[i]<<' ';cout<<'\n';

    // for(int i=KK+1;i<=100;i++) if(i%10==0) cout<<ans<<' ';
    // for(int i=0;i<=10;i++) cout<<running_time[tc][i]<<' ';cout<<'\n';
    // if(!flg) cout<<K+1<<'\n';
    for(int j=1;j<=GA.m;j++) con1[GA.u[j]][GA.v[j]]=con1[GA.v[j]][GA.u[j]]=0;for(int j=1;j<=GB.m;j++) con2[GB.u[j]][GB.v[j]]=con2[GB.v[j]][GB.u[j]]=0;
    // cout<<ans<<' ';
    if(ans!=gr[tc]) E+=1.0*(ans-gr[tc])/gr[tc],e+=ans-gr[tc];//MAE
    else acc++,precision_=1;
    precision+=precision_,num++;
    // cout<<precision_<<' ';


    clock_t end_time=clock();
    // cout<<(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC)<<'\n';

    // if(tc%300==0) cerr<<tc<<'\n';
    // if(gr[tc]) recall+=recall_,prec+=prec_,f1_+=f1,num++;
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    // freopen("use.txt","r",stdin);
    // int n_use;cin>>n_use;while(n_use--){int x;cin>>x;for(int i=(x-1)*20+1;i<=x*20;i++) use[i]=1;}
    // freopen("ground_truth_GED_AIDS.txt","r",stdin);
    int tc=1;
    // cin>>tc;
    // tc=100;
    // for(int i=1;i<=tc;i++) cin>>gr[i];
    // freopen("val_data_GED_AIDS.txt","r",stdin);
    // freopen("case_study_matching_10000.txt","w",stdout);
    // freopen("K_matching_IMDBMulti_70.txt","w",stdout);

    clock_t start_time=clock();

    srand(time(NULL));
    // int tc=1;
    // cin>>tc;
    // tc=10;
    for(int i=1;i<=tc;i++) solve(i);
    
    // for(int i=1;i<=K/100;i++) printf("%.3f ",1.0*bm[i]/10000.0);

    // for(int i=1;i<=K/100;i++) printf("%.3f ",MCS[i]/1000.0);

    // cout<<1.0*e/tc<<' '<<E/tc<<'\n';

    // cout<<1.0*e/num<<' ';//MAE

    // // cout<<prec/num<<' '<<recall/num<<' '<<f1/num<<' ';//prec,recall,f1

    // cout<<1.0*acc/num<<' ';//ACC

    // cout<<precision/num<<' ';//precision

    // clock_t end_time=clock();

    // printf("%.4f\n",(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC));//time
    return 0;
}
