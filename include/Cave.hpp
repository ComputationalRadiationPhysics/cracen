#pragma once
#include <map>       /* map */
#include <set>       /* set */
#include <exception> /* std::out_of_range */
#include <sstream>   /* std::stringstream */
#include <assert.h>  /* assert */
#include <cstddef>    /* nullptr_t */

#include <binaryOperation.hpp> /* op::maximum */
#include <dout.hpp>            /* dout::Dout::getInstance() */

/************************************************************************//**
 * @class Cave
 *
 * @brief A central instance to locate the host 
 *        of vertices.
 *
 * @remark A peer can host several vertices.
 * @todo   Think of vertices hosted by several peers
 *         for fault tolerance purpose
 * @todo remove T_Graph template
 *
 *
 ***************************************************************************/
namespace graybat {

    template <typename T_CAL, typename T_Graph>
    struct Cave {
	typedef T_CAL                            CAL;
	typedef T_Graph                          Graph;
	typedef typename Graph::Vertex           Vertex;
	typedef typename Graph::Edge             Edge;
	typedef typename Graph::GraphID          GraphID;
	typedef typename Graph::EdgeDescription  EdgeDescription;
	typedef typename Graph::GraphDescription GraphDescription;
	typedef typename Vertex::ID              VertexID;
	typedef typename CAL::VAddr              VAddr;
	typedef typename CAL::Event              Event;
	typedef typename CAL::Context            Context;
	typedef typename CAL::ContextID          ContextID;

	// Member
	CAL cal;
	Graph graph;
	std::vector<Vertex> hostedVertices;

	template <class T_Functor>
	Cave(T_Functor graphFunctor) : graph(Graph(graphFunctor())){

	}
	
	/***************************************************************************
	 *
	 * GRAPH OPERATIONS
	 *
	 ***************************************************************************/
	std::vector<Vertex> getVertices(){
	    return graph.getVertices();
	    
	}
	
	Vertex getVertex(VertexID vertexID){
	    return graph.getVertices().at(vertexID);
	    
	}

	std::vector<Vertex> getAdjacentVertices(const Vertex v){
	    return graph.getAdjacentVertices(v);
	    
	}
	
	std::vector<std::pair<Vertex, Edge>> getOutEdges(const Vertex v){
	    return graph.getOutEdges(v);
	    
	}

	std::vector<std::pair<Vertex, Edge>> getInEdges(const Vertex v){
	    return graph.getInEdges(v);
	    
	}


	/***************************************************************************
	 *
	 * MAPPING OPERATIONS
	 *
	 ***************************************************************************/

	/**
	 * @brief Distribution of the graph vertices to the peers of 
	 *        the global context. The distFunctor it the function
	 *        responsible for this distribution.
	 *
	 * @param distFunctor Function for vertex distribution 
	 *                    with the following interface:
	 *                    distFunctor(OwnVAddr, ContextSize, Graph)
	 *
	 */
	template<class T_Functor>
	void distribute(T_Functor distFunctor){
	    hostedVertices = distFunctor(cal.getGlobalContext().getVAddr(), cal.getGlobalContext().size(), graph);
	    announce(graph, hostedVertices);
	}

	
	/**
	 * @brief Announces *vertices* of a *graph* to the network, so that other peers
	 *        know that these *vertices* are hosted by this peer.
	 *
	 * The general workflow includes two steps:
	 *  1. Each peer, that hosts vertices of the *graph* announces its *vertices*
	 *     * Each peer will send its hosted vertices and update its vertices location
	 *     * The host peers will create a new context for *graph*
	 *  2. Vertices can now be located by locateVertex()
	 *  3. use Graphpeer to communicate between vertices
	 *
	 * @remark This is a collective Operation on which either all host peers
	 *         of the supergraph of *graph* have to take part or when *graph* has no
	 *         supergraph then all Communicatos from the globalContext (which should
	 *         be all peers in the network).
	 *
	 * @todo What happens if *vertices* is empty ?
	 * @todo What happens when there already exist an context for *graph* ?
	 * @todo What happens when not all *vertices* of a *graph* were announced ?
	 * @todo Reduce communication from 2 steps (allReduce + allGather) to one
	 *       step (allGatherVar), could reduce communication.
	 *
	 * @param[in] graph  Its vertices will be announced
	 * @param[in] vertices A set of vertices, that will be hosted by this peer
	 *
	 */
	void announce(Graph& graph, const std::vector<Vertex> vertices, const bool global=true){
	    // Get old context from graph
	    Context oldContext = getGraphContext(graph);

	    if(global)
		oldContext = cal.getGlobalContext();

	    if(!oldContext.valid()){
		if(graph.hasSuperGraph()){
		    //std::cout << "hasSuperGraph" << std::endl;
		    oldContext = getGraphContext(graph.superGraph);

		}
		else {
		    //std::cout << "global context" << std::endl;
		    oldContext = cal.getGlobalContext();

		}

	    }
	    else {
		//std::cout << "Has already context" << std::endl;
	    }

	    assert(oldContext.valid());

	    // Create new context for peers which host vertices
	    std::vector<unsigned> hasVertices(1, vertices.size());
	    std::vector<unsigned> recvHasVertices(oldContext.size(), 0);
	    cal.allGather(oldContext, hasVertices, recvHasVertices);

	    std::vector<VAddr> vAddrsWithVertices;

	    for(unsigned i = 0; i < recvHasVertices.size(); ++i){
		if(recvHasVertices[i] > 0){
		    vAddrsWithVertices.push_back(i);
		}
	    }

	    Context newContext = cal.createContext(vAddrsWithVertices, oldContext);
	    graphMap[graph.id] = newContext;
	    // std::cout << "context size: " << newContext.size() << std::endl;
	
	    // Each peer announces the vertices it hosts
	    if(newContext.valid()){

		// Bound graph to new context

	    
		// Retrieve maximum number of vertices per peer
		std::vector<unsigned> myVerticesCount(1,vertices.size());
		std::vector<unsigned> maxVerticesCount(1,  0);
		cal.allReduce(newContext, op::maximum<unsigned>(), myVerticesCount, maxVerticesCount);

		// Gather maxVerticesCount times vertex ids
		std::vector<std::vector<Vertex> > newVertexMaps (newContext.size(), std::vector<Vertex>());
		for(unsigned i = 0; i < maxVerticesCount[0]; ++i){
		    std::vector<int> vertexID(1, -1);
		    std::vector<int> recvData(newContext.size(), 0);

		    if(i < vertices.size()){
			vertexID[0] = graph.getLocalID(vertices.at(i));
		    }

		    cal.allGather(newContext, vertexID, recvData);
		
		   
		    for(unsigned vAddr = 0; vAddr < newVertexMaps.size(); ++vAddr){
			if(recvData[vAddr] != -1){
			    VertexID vertexID = (VertexID) recvData[vAddr];
			    Vertex v = graph.getVertices().at(vertexID);
			    vertexMap[graph.id][v.id] = vAddr;
			    newVertexMaps[vAddr].push_back(v);
		    
			}

		    }
      
		    for(unsigned vAddr = 0; vAddr < newVertexMaps.size(); ++vAddr){
			peerMap[graph.id][vAddr] = newVertexMaps[vAddr];

		    }

		}

	    }

	}

  
	/**
	 * @brief Returns the VAddr of the host peer of *vertex* in the *graph*
	 *
	 * @bug When the location of *vertex* is not known then
	 *      the programm crashes by an exception. 
	 *      This exception should be handled for better
	 *      debugging behaviour.
	 *
	 * @param[in] graph Contains *vertex*.
	 * @param[in] vertex Will be located.
	 *
	 */
	VAddr locateVertex(Graph& graph, Vertex vertex){
	    // std::cerr << graph.id << " " << vertex.id << std::endl;
	    // return 0;
	    return vertexMap.at(graph.id).at(vertex.id);

	}

	/**
	 * @brief Opposite operation of locateVertex(). It returns the
	 *        vertices that are hosted by the peer with
	 *        *vAddr*
	 */
	std::vector<Vertex> getHostedVertices(Graph& graph, VAddr vAddr){
	    return peerMap[graph.id][vAddr];

	}

	/**
	 * @brief Returns the context of the *graph*. All host
	 *        peers of the *graph* are part of the
	 *        returned context.
	 *
	 */
	Context getGraphContext(Graph& graph){
	    return graphMap[graph.id];

	}

	/**
	 * @breif Returns the number of hosts
	 *        of the current **graph**.
	 *
	 */
	unsigned nHosts(){
	    return getGraphContext(graph).size();
	}


	/**
	 * @brief Returns true if the *vertex* is hosted by the
	 *        calling peer otherwise false.
	 *
	 */
	bool peerHostsVertex(Vertex vertex){
	    Context context = getGraphContext(graph);
	    VAddr vaddr     = context.getVAddr();

	    for(Vertex &v : getHostedVertices(graph, vaddr)){
		if(vertex.id == v.id)
		    return true;
	    }
	    return false;
	

	}

	/***************************************************************************
	 *
	 * COMMUNICATION OPERATIONS BASED ON THE GRAPH
	 *
	 ***************************************************************************/


	/**
	 * @brief Synchron transmission of *data* to the *destVertex* on *edge*.
	 *
	 * @param[in] graph The graph in which the communication takes place.
	 * @param[in] destVertex Vertex that will receive the *data*.
	 * @param[in] edge Edge over which the *data* will be transmitted.
	 * @param[in] data Data that will be send.
	 *
	 */
	template <typename T>
	inline void send(const Vertex destVertex, const Edge edge, const T& data){
	    VAddr destVAddr   = locateVertex(graph, destVertex);
	    Context context   = getGraphContext(graph);
	    cal.send(destVAddr, edge.id, context, data);

	}


	/**
	 * @brief Asynchron transmission of *data* to the *destVertex* on *edge*.
	 *
	 * @todo Documentation how data should be formated !!!
	 *
	 * @param[in] graph The graph in which the communication takes place.
	 * @param[in] destVertex Vertex that will receive the *data*.
	 * @param[in] edge Edge over which the *data* will be transmitted.
	 * @param[in] data Data that will be send.
	 *
	 * @return Event Can be waited (Event::wait()) for or checked for (Event::ready())
	 *
	 */
	template <typename T>
	Event asyncSend(const Vertex destVertex, const Edge edge, const T& data){
	    VAddr destVAddr  = locateVertex(graph, destVertex);
	    Context context  = getGraphContext(graph);
	    return cal.asyncSend(destVAddr, edge.id, context, data);
	
	}

	/**
	 * @brief Synchron receive of *data* from the *srcVertex* on *edge*.
	 *
	 * @param[in]  graph The graph in which the communication takes place.
	 * @param[in]  srcVertex Vertex that send the *data*
	 * @param[in]  edge Edge over which the *data* will be transmitted.
	 * @param[out] data Data that will be received
	 *
	 */
	template <typename T>
	inline void recv(const Vertex srcVertex, const Edge edge, T& data){
	    VAddr srcVAddr   = locateVertex(graph, srcVertex);
	    Context context  = getGraphContext(graph);
	    cal.recv(srcVAddr, edge.id, context, data);

	}

	/**
	 * @brief Asynchron receive of *data* from the *srcVertex* on *edge*.
	 *
	 * @param[in]  graph The graph in which the communication takes place.
	 * @param[in]  srcVertex Vertex that send the *data*
	 * @param[in]  edge Edge over which the *data* will be transmitted.
	 * @param[out] data Data that will be received
	 *
	 * @return Event Can be waited (Event::wait()) for or checked for (Event::ready())
	 *
	 */
	template <typename T>
	Event asyncRecv(const Vertex srcVertex, const Edge edge, T& data){
	    VAddr srcVAddr = locateVertex(graph, srcVertex);
	    Context context  = getGraphContext(graph);
	    return cal.asyncRecv(srcVAddr, edge.id, context, data);

	}

	/**************************************************************************
	 *
	 * COLLECTIVE GRAPH OPERATIONS
	 *
	 **************************************************************************/ 

	template <typename T_Data, typename Op>
	void reduce(const Vertex rootVertex, const Vertex srcVertex, Op op, const std::vector<T_Data> sendData, std::vector<T_Data>& recvData){
	    static std::vector<T_Data> reduce;
	    static std::vector<T_Data>* rootRecvData;
	    static unsigned vertexCount = 0;
	    static bool hasRootVertex = false;


	    VAddr rootVAddr   = locateVertex(graph, rootVertex);
	    VAddr srcVAddr    = locateVertex(graph, srcVertex);
	    Context context   = getGraphContext(graph);
	    std::vector<Vertex> vertices = getHostedVertices(graph, srcVAddr);

	    vertexCount++;

	    if(reduce.empty()){
		reduce = std::vector<T_Data>(sendData.size(), 0);
	    }

	    // Reduce locally
	    std::transform(reduce.begin(), reduce.end(), sendData.begin(), reduce.begin(), op);

	    // Remember pointer of recvData from rootVertex
	    if(rootVertex.id == srcVertex.id){
		hasRootVertex = true;
		rootRecvData = &recvData;
	    }

	    // Finally start reduction
	    if(vertexCount == vertices.size()){

		if(hasRootVertex){
		    cal.reduce(rootVAddr, context, op, reduce, *rootRecvData);
		}
		else{
		    cal.reduce(rootVAddr, context, op, reduce, recvData);

		}

		reduce.clear();
		vertexCount = 0;
	    }
	    assert(vertexCount <= vertices.size());

	}

	template <typename T_Data, typename T_Recv, typename Op>
	void allReduce(const Vertex srcVertex, Op op, const std::vector<T_Data> sendData, T_Recv& recvData){

	    static std::vector<T_Data> reduce;
	    static unsigned vertexCount = 0;
	    static std::vector<T_Recv*> recvDatas;

	    VAddr srcVAddr    = locateVertex(graph, srcVertex);
	    Context context   = getGraphContext(graph);
	    std::vector<Vertex> vertices = getHostedVertices(graph, srcVAddr);

	    recvDatas.push_back(&recvData);
	
	    vertexCount++;

	    if(reduce.empty()){
		reduce = std::vector<T_Data>(sendData.size(), 0);
	    }

	    // Reduce locally
	    std::transform(reduce.begin(), reduce.end(), sendData.begin(), reduce.begin(), op);

	    // Finally start reduction
	    if(vertexCount == vertices.size()){

		cal.allReduce(context, op, reduce, *(recvDatas[0]));

		// Distribute Received Data to Hosted Vertices
		for(unsigned i = 1; i < recvDatas.size(); ++i){
		    std::copy(recvDatas[0]->begin(), recvDatas[0]->end(), recvDatas[i]->begin());

		}
	    

		reduce.clear();
		vertexCount = 0;
	    }
	    assert(vertexCount <= vertices.size());

	}
    
      // This function is the hell
      // TODO: Simplify !!!
      // TODO: Better software design required !!!
      template <typename T_Send, typename T_Recv>
      void gather(const Vertex rootVertex, const Vertex srcVertex, const T_Send sendData, T_Recv& recvData, const bool reorder){
	typedef typename T_Send::value_type SendValueType;
	typedef typename T_Recv::value_type RecvValueType;
	
	static std::vector<SendValueType> gather;
	static T_Recv* rootRecvData     = NULL;
	static bool peerHostsRootVertex = false;
	static unsigned nGatherCalls    = 0;

	nGatherCalls++;

	VAddr rootVAddr  = locateVertex(graph, rootVertex);
	Context context  = getGraphContext(graph);

	// Insert data of srcVertex to the end of the gather vector
	gather.insert(gather.end(), sendData.begin(), sendData.end());

	// Store recv pointer of rootVertex
	if(srcVertex.id == rootVertex.id){
	  rootRecvData = &recvData;
	  peerHostsRootVertex = true;
	}

	if(nGatherCalls == hostedVertices.size()){
	  std::vector<unsigned> recvCount;
	  std::vector<unsigned> prefixsum(context.size(),0);

	  if(peerHostsRootVertex){
	    cal.gatherVar(rootVAddr, context, gather, *rootRecvData, recvCount);

	    // TODO
	    // std::partial_sum might do the job
	    unsigned sum = 0;
	    for(unsigned count_i = 0; count_i < recvCount.size(); ++count_i){
	      prefixsum[count_i] = sum;
	      sum += recvCount[count_i];
	    }
		    
	    // Reordering code
	    if(reorder){
	      std::vector<RecvValueType> recvDataReordered(recvData.size());
	      for(unsigned vAddr = 0; vAddr < context.size(); vAddr++){
		std::vector<Vertex> hostedVertices = getHostedVertices(graph, vAddr);
		unsigned nElementsPerVertex = recvCount.at(vAddr) / hostedVertices.size();

		unsigned hVertex_i=0;
		for(Vertex v: hostedVertices){

		  std::copy(rootRecvData->begin()+(prefixsum[vAddr] + (hVertex_i * nElementsPerVertex)),
			    rootRecvData->begin()+(prefixsum[vAddr] + (hVertex_i * nElementsPerVertex)) + (nElementsPerVertex),
			    recvDataReordered.begin()+(v.id*nElementsPerVertex));
		  hVertex_i++;

		}
			    
	      }
	      std::copy(recvDataReordered.begin(), recvDataReordered.end(), rootRecvData->begin());

	    }
		
	  }
	  else {
	    cal.gatherVar(rootVAddr, context, gather, recvData, recvCount);
	  }
	    
	  gather.clear();
	  nGatherCalls = 0;

	}

      }


	/**
	 *  @todo get rid of sendData.begin(), sendData.end()
	 *        T_Data should only use .size() and .data()
	 *
	 **/
	template <typename T_Send, typename T_Recv>
	void allGather(const Vertex srcVertex, T_Send sendData, T_Recv& recvData, const bool reorder){
	    typedef typename T_Send::value_type T_Send_Container;
	
	    static std::vector<T_Send_Container> gather;
	    static std::vector<T_Recv*> recvDatas;

	    VAddr srcVAddr  = locateVertex(graph, srcVertex);
	    Context context = getGraphContext(graph);
	    std::vector<Vertex> vertices = getHostedVertices(graph, srcVAddr);

	    gather.insert(gather.end(), sendData.begin(), sendData.end());
	    recvDatas.push_back(&recvData);


	    if(gather.size() == vertices.size()){
		std::vector<unsigned> recvCount;

		cal.allGatherVar(context, gather, *(recvDatas[0]), recvCount);

		if(reorder){
		    std::vector<typename T_Recv::value_type> recvDataReordered(recvData.size());
		    unsigned vAddr = 0;
		    for(unsigned recv_i = 0; recv_i < recvData.size(); ){
			std::vector<Vertex> hostedVertices = getHostedVertices(graph, vAddr);
			for(Vertex v: hostedVertices){
			    recvDataReordered.at(v.id) = recvDatas[0]->data()[recv_i];
			    recv_i++;
			}
			vAddr++;
		    }
		    for(unsigned i = 0; i < recvDataReordered.size(); ++i){
			recvDatas[0]->data()[i] = recvDataReordered[i];
		    }
		    
		}

		// Distribute Received Data to Hosted Vertices
		//unsigned nElements = std::accumulate(recvCount.begin(), recvCount.end(), 0);
		for(unsigned i = 1; i < recvDatas.size(); ++i){
		    std::copy(recvDatas[0]->begin(), recvDatas[0]->end(), recvDatas[i]->begin());

		}
	    
		gather.clear();


	    }

	}

	void synchronize(Graph &graph){
	    Context context = getGraphContext(graph);
	    cal.synchronize(context);

	}

    
    private:

	/***************************************************************************
	 *
	 * MAPS
	 *
	 ***************************************************************************/
	// Maps vertices to its hosts
	std::map<GraphID, std::map<VertexID, VAddr> > vertexMap;
    
	// Each graph is mapped to a context of the peer
	std::map<GraphID, Context> graphMap; 

	// List of vertices a peer hosts of a graph
	std::map<GraphID, std::map<VAddr, std::vector<Vertex>> > peerMap;

	// Count of vertices that are in a collective operation in the moment
	std::map<GraphID, std::map<VertexID, unsigned>> vertexCount;


	/***************************************************************************
	 *
	 * AUXILIARY FUNCTIONS
	 *
	 ***************************************************************************/

	/**
	 * @brief Returns a set of all host peer VAddrs of the *graph*
	 *
	 */
	std::vector<VAddr> getGraphHostVAddrs(Graph& graph){
	    std::vector<Vertex> vertices = graph.getVertices();

	    std::set<VAddr> vAddrs;
	    for(Vertex vertex : vertices){
		vAddrs.insert(locateVertex(graph, vertex));
	    }

	    return std::vector<VAddr>(vAddrs.begin(), vAddrs.end());

	}

	std::string generateID(Graph &graph, Vertex vertex){
	    std::stringstream idSS; 
	    if(vertexCount[graph.id][vertex.id] > 0){
		throw std::logic_error("Can not perform collective operation on same vertex in same graph simultaneous");
	    }
	    else {
		idSS << graph.id << vertexCount[graph.id][vertex.id]++;
		return idSS.str();
	    }

	}


	/**
	 * @brief Creates a context from the given *subgraph* inherited from
	 *        the context of the given graph.
	 *
	 * @param[in] graph is the supergraph of subGraph
	 * @param[in] subGraph is a subgraph of graph
	 *
	 */
	void createGraphContext(Graph& graph, Graph& subGraph){
	    std::vector<VAddr> vAddrs = getGraphHostVAddrs(subGraph);

	    Context oldContext = getGraphContext(graph);
	    Context newContext = cal.createContext(vAddrs, oldContext);
	    if(newContext.valid()){
		graphMap[subGraph.id] = newContext;
	    }

	}

	/**
	 * @brief Creates a context for the the given graph inherited from
	 *        the global context. After this step, vertices within
	 *        this graph can do communication.
	 *
	 * @param[in] graph for which a new context from global context will be created.
	 */
	void createGraphContext(Graph& graph){
	    std::vector<VAddr> vAddrs = getGraphHostVAddrs(graph);

	    Context oldContext = cal.getGlobalContext();
	    Context newContext = cal.createContext(vAddrs, oldContext);
	
	    graphMap[graph.id] = newContext;
    
	}

    };

} // namespace graybat
