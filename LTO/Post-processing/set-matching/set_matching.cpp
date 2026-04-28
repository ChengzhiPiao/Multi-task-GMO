#pragma GCC optimize(2)
#include<bits/stdc++.h>
#define N 65
#define K 100000
#define K_max 100
#define rand_time 3
#define LL long long
#define inf 214748364777777ll
#define INF 2147483647214748364ll
#define LD long double
#define eps 1e-4
#define max_dif 1000000
#define dif 1000
#define s_id set<int>::iterator
using namespace std;
int ans,n,m,mapping[N][N],pre_mapping[N],f[N],a[N],b[N];LL cost[N][N],c[N][N],c_tmp[N][N];vector<int> A[N],B[N];bool cut[N][N];
LL dis[N<<1],slack[N];int vis[N<<1],vis2[N<<1];bool in[N];/*vector<int> miss;*/
int tot,tt,vx[N],lst[N],ok[N][N],hd,tl,q[N*N*N];// for choosing a random vertex
bool use[K+5];int Ans[K+5][15];double running_time[K+5][15];
int ged[N];LL bm[K+5];
struct graph{int n,m;int labels[13][10005];}GA,GB;bool con1[N][N],con2[N][N];
int MCS[K+5],gr[K+5],num,acc;double E,recall,prec,f1,precision;LL e;double recall_=0,prec_=0,f1_=0,precision_=0;
unordered_map<int,int> mp;int tmp_matching[N];
int P=998244353,mod=1e9+7;
struct path{
    int mapping[N];bool com[N];
    int gen_path(){
        int res=0,attr=0;mp.clear();for(int i=1;i<=GA.n;i++) if(mapping[i]<=GB.n) attr++;
        for(int j=1;j<=GA.m;j++){int s=0;for(int i=1;i<=GA.n;i++) if(mapping[i]<=GB.n) s=(1ll*s*P%mod+GA.labels[i][j])%mod;mp[s]++;}
        for(int j=1;j<=GB.m;j++){int s=0;for(int i=1;i<=GA.n;i++) if(mapping[i]<=GB.n) s=(1ll*s*P%mod+GB.labels[mapping[i]][j])%mod;if(mp[s]) mp[s]--,res++;}
        // for(int j=1;j<=GA.m;j++){string s="";for(int i=1;i<=GA.n;i++) if(mapping[i]<=GB.n) s+=GA.labels[i][j]+" ";mp[s]++;}
        // for(int j=1;j<=GB.m;j++){string s="";for(int i=1;i<=GA.n;i++) if(mapping[i]<=GB.n) s+=GB.labels[mapping[i]][j]+" ";if(mp[s]) mp[s]--,res++;}
        return res*attr;
    }
}gt,pred/*[K_max+5]*/;
int d_nxt[N],d_lst[N];
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
        int x=best_matching[S]+n;
        for(int i=1;i<=m;i++) d_nxt[i]=i+1,d_lst[i]=i-1;d_nxt[0]=1,d_lst[m+1]=m;
        for(int i=1;i<=m;i++) if(vis[i+n]) d_nxt[d_lst[i]]=d_nxt[i],d_lst[d_nxt[i]]=d_lst[i];
        for(int i=1;i<=n+m;i++) dis[i]=INF;dis[S]=0;while(1){
            vis[S]=1;if(S>n) d_nxt[d_lst[S-n]]=d_nxt[S-n],d_lst[d_nxt[S-n]]=d_lst[S-n];
            if(S<=n){for(int i=d_nxt[0];i<=m;i=d_nxt[i]) if(i!=best_matching[S]&&dis[S]+c[S][i]<dis[i+n]&&c[S][i]!=inf) dis[i+n]=dis[S]+c[S][i],lst[i]=S;}
            else{if(match[S-n]){dis[match[S-n]]=dis[S],S=match[S-n],vis[S]=1;continue;}} 
            S=0;for(int i=d_nxt[0];i<=m;i=d_nxt[i]) if(!vis[i+n]&&dis[i+n]!=INF) if(!S||dis[i+n]<dis[S]) S=i+n;if(!S) break;vis[S]=1;
            if(S==x) break;
        }
    }
    inline int get_upper_bound(){
        int res=0,attr=0;mp.clear();for(auto i:I) if(i.first<=GA.n&&i.second>GB.n) attr++;
        for(int j=1;j<=GA.m;j++){int s=0;for(auto i:I) if(i.first<=GA.n&&i.second<=GB.n) s=(1ll*s*P%mod+GA.labels[i.first][j])%mod;mp[s]++;}
        for(int j=1;j<=GB.m;j++){int s=0;for(auto i:I) if(i.first<=GA.n&&i.second<=GB.n) s=(1ll*s*P%mod+GB.labels[i.second][j])%mod;if(mp[s]) mp[s]--,res++;}
        return res*(GA.n-attr);
    }
    inline void get_ans_sec(){
        for(int j=1;j<=n;j++) pred.mapping[j]=second_matching[j];int tmp=pred.gen_path();ans=max(ans,tmp);//bound=max(bound,tmp);
        int res=0;for(int j=1;j<=GA.n;j++) if(mapping[j][min(second_matching[j],GB.n+1)]) res++;
        if(ans==tmp){precision_=1.0*res/GA.n;for(int j=1;j<=GA.n;j++) pre_mapping[j]=second_matching[j];}
    }
    inline void get_ans(){
        for(int j=1;j<=n;j++) pred.mapping[j]=best_matching[j];int tmp=pred.gen_path();ans=max(ans,tmp);//bound=max(bound,tmp);
        int res=0;for(int j=1;j<=GA.n;j++) if(mapping[j][min(best_matching[j],GB.n+1)]) res++;
        if(ans==tmp){precision_=1.0*res/GA.n;for(int j=1;j<=GA.n;j++) pre_mapping[j]=best_matching[j];}
    }
    inline void get_second(){
        memset(second_matching,0,sizeof(second_matching)),memset(match,0,sizeof(match)),memset(vis,0,sizeof(vis));
        for(int i=0;i<I.size();i++) vis[I[i].first]=vis[I[i].second+n]=2;
        get_h();for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) if(best_matching[i]==j) c[i][j]=0,match[j]=i;else c[i][j]=h[i]+h[j+n]-cost[i][j];
        for(int i=0;i<O.size();i++) c[O[i].first][O[i].second]=inf;
        tot=0;for(int i=1;i<=n;i++) if(!vis[i]) vx[++tot]=i;if(!tot){sm=-INF;return;} 
        for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) cut[i][j]=0;for(int i=1;i<=n+m;i++) vis2[i]=vis[i];
        V=0,sm=-INF;for(int i=1;i<=tot;i++){
            int S=vx[i],x=best_matching[S],sa=a[S],sb=b[x];if(cut[sa][sb]) continue;cut[sa][sb]=1;
            for(int ii=1;ii<=n+m;ii++) vis[ii]=vis2[ii];
            for(int ii=0;ii<A[sa].size();ii++) for(int jj=0;jj<B[sb].size();jj++){int k=A[sa][ii],j=B[sb][jj];if(best_matching[k]^j) c_tmp[k][j]=c[k][j],c[k][j]=inf;}
            dijkstra(S);if(dis[x+n]<INF&&bm-dis[x+n]>sm){
                for(int j=1;j<=n;j++) tmp_matching[j]=second_matching[j];LL tmp_sm=sm;int tmp_V=V;
                sm=bm-dis[x+n],V=S;
                for(int j=1;j<=n;j++) second_matching[j]=best_matching[j];
                while(1){second_matching[lst[x]]=x;if(lst[x]==S) break;x=best_matching[lst[x]];} // find the cycle and get the second-best-matching
                // get_ans_sec();
                if(sm<=tmp_sm){for(int j=1;j<=n;j++) second_matching[j]=tmp_matching[j];sm=tmp_sm,V=tmp_V;}
            }
            for(int ii=0;ii<A[sa].size();ii++) for(int jj=0;jj<B[sb].size();jj++){int k=A[sa][ii],j=B[sb][jj];if(best_matching[k]^j) c[k][j]=c_tmp[k][j];}
        }
        get_ans_sec();
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
inline void read(){
    cin>>n>>m;for(int i=1;i<=n;i++){double x;for(int j=1;j<=m;j++) cin>>x,cost[i][j]=(LL)x;} 
    for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) cin>>mapping[i][j];
    cin>>GA.n>>GA.m;for(int i=1;i<=GA.n;i++) for(int j=1;j<=GA.m;j++) cin>>GA.labels[i][j];
    cin>>GB.n>>GB.m;for(int i=1;i<=GB.n;i++) for(int j=1;j<=GB.m;j++) cin>>GB.labels[i][j];
}
inline int gf(int x){return f[x]==x?x:f[x]=gf(f[x]);}
vector<int> NB[N<<1],NB2[N<<1];
inline void init(){
    memset(vis,0,sizeof(vis));G[1].I.clear(),G[1].O.clear();

    //complete the cost matrix
    for(int i=1;i<=GA.n;i++) for(int j=1;j<=GB.n;j++){
        mp.clear();bool flg=0;for(int k=1;k<=GA.m;k++) mp[GA.labels[i][k]]++;
        for(int k=1;k<=GB.m;k++) if(mp[GB.labels[j][k]]){flg=1;break;} if(!flg) cost[i][j]=0;
    }
    for(int i=1;i<=n;i++) for(int j=m+1;j<=n+m-1;j++) cost[i][j]=cost[i][m];m=n+m-1;
    for(int i=n+1;i<=m;i++) for(int j=1;j<=m;j++) cost[i][j]=0;n=m;

    //merge the set
    for(int i=1;i<=GB.n;i++) f[i]=i;for(int i=GB.n+1;i<=n;i++) f[i]=GB.n+1;// column
    // for(int i=1;i<=GB.n;i++) for(int j=i+1;j<=GB.n;j++) if(gf(i)^gf(j)){bool flg=1;for(int k=1;k<=n;k++) if(abs(cost[k][i]-cost[k][j])>dif){flg=0;break;} if(flg){for(int k=1;k<=n;k++) cost[k][j]=cost[k][i];f[gf(j)]=gf(i);}} //case 2: cost[i][k]=cost[j][k]
    // for(int i=1;i<=GA.n;i++){LL res=0;int cnt=0;for(int j=GB.n+1;j<=n;j++){res+=cost[i][j];if(cost[i][j]) cnt++;}res/=(n-GB.n);for(int j=GB.n+1;j<=n;j++) cost[i][j]=res;}
    // for(int i=1;i<=GA.n;i++){LL res=0;int cnt=0;for(int j=GB.n+1;j<=n;j++){res+=cost[i][j];if(cost[i][j]) cnt++;}res/=max(cnt,1);for(int j=GB.n+1;j<=n;j++) cost[i][j]=res;}
    // tot=0;for(int i=1;i<=n;i++) if(gf(i)==i){tot++,B[tot].clear();for(int j=i;j<=n;j++) if(gf(j)==i) b[j]=tot,B[tot].push_back(j);} 
    // for(int i=1;i<=GB.n;i++) for(int j=i+1;j<=GB.n;j++) if(gf(i)^gf(j)) if((NB[i+GA.n]==NB[j+GA.n]||NB2[i+GA.n]==NB2[j+GA.n])&&GB.labels[i]==GB.labels[j]) f[gf(j)]=gf(i);
    for(int i=1;i<=GB.n;i++) for(int j=i+1;j<=GB.n;j++) if(gf(i)^gf(j)){bool flg=1;for(int k=1;k<=n;k++) if(abs(cost[k][i]-cost[k][j])>dif){flg=0;break;} if(flg){for(int k=1;k<=n;k++) cost[k][j]=cost[k][i];f[gf(j)]=gf(i);}} //case 2: cost[i][k]=cost[j][k]
    tot=0;for(int i=1;i<=n;i++) if(gf(i)==i){tot++,B[tot].clear();for(int j=i;j<=n;j++) if(gf(j)==i){b[j]=tot,B[tot].push_back(j);for(int k=1;k<=n;k++) cost[k][j]=cost[k][i];}}
    for(int i=1;i<=GA.n;i++) f[i]=i;for(int i=GA.n+1;i<=n;i++) f[i]=GA.n+1;// row
    // for(int i=1;i<=n;i++) for(int j=i+1;j<=n;j++) if(gf(i)^gf(j)){bool flg=1;for(int k=1;k<=n;k++) if(abs(cost[i][k]-cost[j][k])>dif){flg=0;break;} if(flg){for(int k=1;k<=n;k++) cost[j][k]=cost[i][k];f[gf(j)]=gf(i);}} //case 2: cost[i][k]=cost[j][k]
    // tot=0;for(int i=1;i<=n;i++) if(gf(i)==i){tot++,A[tot].clear();for(int j=i;j<=n;j++) if(gf(j)==i) a[j]=tot,A[tot].push_back(j);}
    // for(int i=1;i<=GA.n;i++) for(int j=i+1;j<=GA.n;j++) if(gf(i)^gf(j)) if((NB[i]==NB[j]||NB2[i]==NB2[j])&&GA.labels[i]==GA.labels[j]) f[gf(j)]=gf(i);
    for(int i=1;i<=GA.n;i++) for(int j=i+1;j<=GA.n;j++) if(gf(i)^gf(j)){bool flg=1;for(int k=1;k<=n;k++) if(abs(cost[i][k]-cost[j][k])>dif){flg=0;break;} if(flg){for(int k=1;k<=n;k++) cost[j][k]=cost[i][k];f[gf(j)]=gf(i);}} //case 2: cost[i][k]=cost[j][k]
    tot=0;for(int i=1;i<=n;i++) if(gf(i)==i){tot++,A[tot].clear();for(int j=i;j<=n;j++) if(gf(j)==i){a[j]=tot,A[tot].push_back(j);for(int k=1;k<=n;k++) cost[j][k]=cost[i][k];}}
    // for(int i=1;i<=tot;i++) cout<<A[i].size()<<' ';cout<<'\n';
    tot=0,G[1].KM();
}
int KK;
inline void AKM(int tc){
    clock_t start_time=clock();
    bool flg=1;G[1].print();G[1].get_ans(),G[1].get_second();// Get the second-best matching
    for(int i=0;i<=10;i++) Ans[tc][i]=888888,running_time[tc][i]=0;
    Ans[tc][0]=ans;
    clock_t end_time=clock();
    running_time[tc][0]=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
    int tmp=ans;
    for(int k=2;k<=K_max;k++){
        int id=1;for(int i=2;i<k;i++) if(G[i].sm>G[id].sm) id=i;// find the maximum second_best_matching
        if(G[id].sm==-INF){KK=k-1;/*cout<<"Only "<<k-1<<" matchings exist"<<'\n'<<'\n';*/flg=0;for(int i=k;i<=K_max;i++) G[i].bm=-INF;/*for(int i=1;i<=n;cout<<'\n',i++) for(int j=1;j<=n;j++) cout<<cost[i][j]<<' ';*/break;}
        int S=G[id].V,x=G[id].best_matching[S],sa=a[S],sb=b[x];G[k].I=G[id].I,G[k].O=G[id].O,G[k].O.push_back(make_pair(S,x));
        for(int ii=0;ii<A[sa].size();ii++) for(int jj=0;jj<B[sb].size();jj++)
        {int i=A[sa][ii],j=B[sb][jj];if(G[id].best_matching[i]==j) G[id].I.push_back(make_pair(i,j));else G[k].O.push_back(make_pair(i,j));}
        memcpy(G[k].best_matching,G[id].second_matching,sizeof(G[id].second_matching)),G[k].bm=G[id].sm,G[k].print();
        G[id].get_second(),G[k].get_second();// update I&O and second_best_matching
        if(G[id].get_upper_bound()<=ans) G[id].sm=-INF;if(G[k].get_upper_bound()<=ans) G[k].sm=-INF;
        if(k%10==0){
            end_time=clock();
            Ans[tc][k/10]=ans;
            running_time[tc][k/10]=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
        }
    }
    end_time=clock();
    for(int i=1;i<=10;i++){
        if(i*10>KK) running_time[tc][i]=max(running_time[tc][i],(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC)),Ans[tc][i]=ans;
    }
    // if(flg) cout<<K<<"-best matchings are found"<<'\n'<<'\n';
}
// int MCS[K+5],deg[N],gr[K+5],rand_v[N],rk[N],num,acc;bool del[N];double E,recall,prec,f1,precision;LL e;

// inline bool cmp2(int x,int y){return rand_v[x]<rand_v[y];}
double tot_time,rat;
inline void solve(int tc){
    ans=0;precision_=0;
    KK=K_max;//cout<<tc<<'\n';
    // cout<<"Test data "<<tc<<":\n";
    read();// read the data

    
    // if(tc%100==0) cerr<<tc<<'\n';
    // return;

    // if(!use[tc]) return;
    clock_t start_time=clock();
    init();// Get the matching matrix and the best matching
    AKM(tc);// Algorithm of K-best Matching
    // for(int i=1;i<=K;i++) bm[(i-1)/100+1]+=G[i].bm,cout<<G[i].bm<<' ';cout<<'\n';
    // bool flg=0;double recall_=0,prec_=0,f1_=0,precision_=0;
    // for(int j=1;j<=GA.n;j++) gt.mapping[j]=mapping[j];gt.gen_path();
    // for(int i=1;i<=KK;i++){
    //     for(int j=1;j<=GA.n;j++) pred/*[i]*/.mapping[j]=G[i].best_matching[j];int tmp=pred/*[i]*/.gen_path();ans=max(ans,tmp);
    //     int res=0;for(int j=1;j<=GA.n;j++) if(mapping[j][min(G[i].best_matching[j],GB.n+1)]) res++;precision_=max(precision_,1.0*res/GA.n);
    //     // int intersec=0;for(int j=1;j<=GA.n;j++) if(pred[i].com[j]&&gt.com[j]) intersec++;
    //     // if(!intersec) continue;recall_=max(recall_,1.0*intersec/gr[tc]),prec_=max(prec_,1.0*intersec/tmp),f1_=max(f1_,2.0/(1.0*gr[tc]/intersec+1.0*tmp/intersec));
    //     // if(i%100==0) MCS[i/100]+=ans;
    //     // if(ans==gr[tc]){flg=1,cout<<i<<'\n';break;}
    // }
    // if(!flg) cout<<K+1<<'\n';
    // if(tc%100==0) cerr<<tc<<'\n';
    // cout<<ans<<'\n';
    // for(int i=0;i<=10;i++) cout<<Ans[tc][i]<<' ';
    // for(int i=0;i<=10;i++) cout<<running_time[tc][i]<<' ';
    // cout<<'\n';

    cout<<"LTO = "<<ans<<'\n';
    cout<<"Row Matching = ";for(int i=1;i<=GA.n;i++) if(pre_mapping[i]<=GB.n) cout<<"("<<i-1<<","<<pre_mapping[i]-1<<") ";cout<<'\n';
    
    if(ans!=gr[tc]) E+=1.0*(gr[tc]-ans)/min(GA.n*GA.m,GB.n*GB.m)/*gr[tc]*/,e+=gr[tc]-ans;else acc++,precision_=1;
    precision+=precision_,num++;

    clock_t end_time=clock();
    tot_time+=(double)(1.0*(end_time-start_time)/CLOCKS_PER_SEC);
    // recall+=recall_,prec+=prec_,f1_+=f1;
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    // freopen("use.txt","r",stdin);
    // int n_use;cin>>n_use;while(n_use--){int x;cin>>x;for(int i=(x-1)*20+1;i<=x*20;i++) use[i]=1;}
    // freopen("ground_truth_OVERLAP_ranking_git.txt","r",stdin);
    int tc=1;
    // cin>>tc;
    // for(int i=1;i<=tc;i++) cin>>gr[i],rat+=gr[i];
    // freopen("val_data_OVERLAP_ranking_git.txt","r",stdin);
    // freopen("K_set_matching_ub_ranking_git_v3.txt","w",stdout);

    

    // srand(time(NULL));
    // cin>>tc;
    // tc=1;
    // tc=200;
    for(int i=1;i<=tc;i++) solve(i);

    // cout<<(int)rat<<'\n';
    
    // for(int i=1;i<=K/100;i++) printf("%.3f ",1.0*bm[i]/10000.0);

    // for(int i=1;i<=K/100;i++) printf("%.3f ",MCS[i]/1000.0);
    
    // cout<<1.0*e/tc<<' '<<E/tc<<'\n';

    // cout<<1.0*E/num<<' '<<1.0*e/num<<' ';//MAE

    // cout<<1.0*acc/num<<'\n';//ACC

    // // cout<<prec/num<<' '<<recall/num<<' '<<f1/num<<' ';//prec,recall,f1

    // cout<<precision/num<<' ';//precision

    // printf("%.4f\n",tot_time);//time
    return 0;
}
