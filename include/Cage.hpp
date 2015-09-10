#pragma once
#include <map>       /* map */
#include <set>       /* set */
#include <exception> /* std::out_of_range */
#include <sstream>   /* std::stringstream */
#include <assert.h>  /* assert */
#include <cstddef>    /* nullptr_t */
#include <algorithm> /* std::max */

#include <dout.hpp>            /* dout::Dout::getInstance() */
#include <utils.hpp> /* exclusivePrefixSum */

namespace graybat {

    /************************************************************************//**
     * @class Cage
     *
     * @brief The Communication And Graph Environment enables to communicate
     *        on basis of a graph with methods of a user defined communication
     *	      library.
     *
     * A cage is defined by its Communication and Graph policy. The communication
     * policy provides methods for point to point and collective operations.
     * The graph policy provides methods to query graph imformation of the
     * cage graph. 
     *
     * @remark A peer can host several vertices.
     *
     *
     ***************************************************************************/
    template <typename T_CommunicationPolicy, typename T_GraphPolicy>
    struct Cage {
	typedef T_CommunicationPolicy                   CommunicationPolicy;
        typedef T_GraphPolicy                           GraphPolicy;
	typedef typename GraphPolicy::Vertex            Vertex;
	typedef typename GraphPolicy::Edge              Edge;
	typedef typename GraphPolicy::GraphID           GraphID;
	typedef typename GraphPolicy::EdgeDescription   EdgeDescription;
	typedef typename GraphPolicy::GraphDescription  GraphDescription;
	typedef typename Vertex::ID                     VertexID;
	typedef typename CommunicationPolicy::VAddr     VAddr;
	typedef typename CommunicationPolicy::Event     Event;
	typedef typename CommunicationPolicy::Context   Context;
	typedef typename CommunicationPolicy::ContextID ContextID;


	template <class T_Functor>
	Cage(T_Functor graphFunctor) : graph(GraphPolicy(graphFunctor())){

	}

      	/***************************************************************************
	 *
	 * MEMBER
	 *
	 ***************************************************************************/
	CommunicationPolicy comm;
        GraphPolicy         graph;
        Context             graphContext;
	std::vector<Vertex> hostedVertices;


      	/***************************************************************************
	 *
	 * MAPS
	 *
	 ***************************************************************************/
	// Maps vertices to its hosts
	std::map<VertexID, VAddr> vertexMap;
    
	// Each graph is mapped to a context of the peer
	std::map<GraphID, Context> graphMap; 

	// List of vertices of the hosts
	std::map<VAddr, std::vector<Vertex> > peerMap;

	
	/***********************************************************************//**
         *
	 * @name Graph Operations
	 * 
	 * @{
	 *
	 ***************************************************************************/
	std::vector<Vertex> getVertices(){
	    return graph.getVertices();
	    
	}
	
	Vertex getVertex(const VertexID vertexID){
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
	/** @} */

	/***********************************************************************//**
	 *
	 * @name Mapping Operations
	 *
	 * @{
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
	    hostedVertices = distFunctor(comm.getGlobalContext().getVAddr(), comm.getGlobalContext().size(), graph);
	    announce(hostedVertices);
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
	void announce(const std::vector<Vertex> vertices, const bool global=true){
	    // Get old context from graph
    	  Context oldContext = graphContext;

	  if(global){
		oldContext = comm.getGlobalContext();

	  }
		
	    if(!oldContext.valid()){

	    }
	    else {
		//std::cout << "Has already context" << std::endl;
	    }

	    assert(oldContext.valid());

	    // Create new context for peers which host vertices
	    std::vector<unsigned> nVertices(1, vertices.size());
	    std::vector<unsigned> recvHasVertices(oldContext.size(), 0);
	    comm.allGather(oldContext, nVertices, recvHasVertices);

	    std::vector<VAddr> vAddrsWithVertices;

	    for(unsigned i = 0; i < recvHasVertices.size(); ++i){
		if(recvHasVertices[i] > 0){
		    vAddrsWithVertices.push_back(i);
		}
	    }

	    Context newContext = comm.createContext(vAddrsWithVertices, oldContext);
	    graphContext = newContext;
	
	    // Each peer announces the vertices it hosts
	    if(newContext.valid()){
		// Bound graph to new context

	    
		// Retrieve maximum number of vertices per peer
		std::vector<unsigned> nVertices(1,vertices.size());
		std::vector<unsigned> maxVerticesCount(1,  0);
		comm.allReduce(newContext, maximum<unsigned>(), nVertices, maxVerticesCount);

		// Gather maxVerticesCount times vertex ids
		std::vector<std::vector<Vertex> > newVertexMaps (newContext.size(), std::vector<Vertex>());
		for(unsigned i = 0; i < maxVerticesCount[0]; ++i){
		    std::vector<int> vertexID(1, -1);
		    std::vector<int> recvData(newContext.size(), 0);

		    if(i < vertices.size()){
			vertexID[0] = graph.getLocalID(vertices.at(i));
		    }

		    comm.allGather(newContext, vertexID, recvData);
		
		   
		    for(unsigned vAddr = 0; vAddr < newVertexMaps.size(); ++vAddr){
			if(recvData[vAddr] != -1){
			    VertexID vertexID = (VertexID) recvData[vAddr];
			    Vertex v = graph.getVertices().at(vertexID);
			    vertexMap[v.id] = vAddr;
			    newVertexMaps[vAddr].push_back(v);
		    
			}

		    }
      
		    for(unsigned vAddr = 0; vAddr < newVertexMaps.size(); ++vAddr){
			peerMap[vAddr] = newVertexMaps[vAddr];

		    }

		}

	    }

	}

	template <typename T>
	struct maximum {

	    T operator()(const T a, const T b){
		return std::max(a, b);
	    }
	    
	};

  
	/**
	 * @brief Returns the VAddr of the host of *vertex* in the graph
	 *
	 * @bug When the location of *vertex* is not known then
	 *      the programm crashes by an exception. 
	 *      This exception should be handled for better
	 *      debugging behaviour.
	 *
	 * @param[in] vertex Will be located.
	 *
	 */
	VAddr locateVertex(Vertex vertex){
	    return vertexMap.at(vertex.id);

	}

	/**
	 * @brief Opposite operation of locateVertex(). It returns the
	 *        vertices that are hosted by the peer with
	 *        *vAddr*
	 */
	std::vector<Vertex> getHostedVertices(const VAddr vAddr){
	  return peerMap[vAddr];

	}

	/**
	 * @brief Returns true if the *vertex* is hosted by the
	 *        calling peer otherwise false.
	 *
	 */
	bool peerHostsVertex(Vertex vertex){
	    VAddr vaddr     = graphContext.getVAddr();

	    for(Vertex &v : getHostedVertices(vaddr)){
		if(vertex.id == v.id)
		    return true;
	    }
	    return false;
	

	}

	/** @} */

	/***********************************************************************//**
	 *
	 * @name Point to Point Communication Operations 
	 *
	 * @{
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
	    VAddr destVAddr   = locateVertex(destVertex);
	    comm.send(destVAddr, edge.id, graphContext, data);

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
	    VAddr destVAddr  = locateVertex(destVertex);
	    return comm.asyncSend(destVAddr, edge.id, graphContext, data);
	
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
	    VAddr srcVAddr   = locateVertex(srcVertex);
	    comm.recv(srcVAddr, edge.id, graphContext, data);

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
	Event asypncRecv(const Vertex srcVertex, const Edge edge, T& data){
	    VAddr srcVAddr = locateVertex(srcVertex);
	    return comm.asyncRecv(srcVAddr, edge.id, graphContext, data);

	}

	/** @} */

	/**********************************************************************//**
	 *
	 * @name Collective Communication Operations 
	 *
	 * @{
	 *
	 **************************************************************************/ 

	template <typename T_Data, typename Op>
	void reduce(const Vertex rootVertex, const Vertex srcVertex, Op op, const std::vector<T_Data> sendData, std::vector<T_Data>& recvData){
	    static std::vector<T_Data> reduce;
	    static std::vector<T_Data>* rootRecvData;
	    static unsigned vertexCount = 0;
	    static bool hasRootVertex = false;


	    VAddr rootVAddr   = locateVertex(rootVertex);
	    VAddr srcVAddr    = locateVertex(srcVertex);
	    Context context   = graphContext;
	    std::vector<Vertex> vertices = getHostedVertices(srcVAddr);

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
		    comm.reduce(rootVAddr, context, op, reduce, *rootRecvData);
		}
		else{
		    comm.reduce(rootVAddr, context, op, reduce, recvData);

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

	    VAddr srcVAddr    = locateVertex(srcVertex);
	    Context context   = graphContext;
	    std::vector<Vertex> vertices = getHostedVertices(srcVAddr);

	    recvDatas.push_back(&recvData);
	
	    vertexCount++;

	    if(reduce.empty()){
		reduce = std::vector<T_Data>(sendData.size(), 0);
	    }

	    // Reduce locally
	    std::transform(reduce.begin(), reduce.end(), sendData.begin(), reduce.begin(), op);

	    // Finally start reduction
	    if(vertexCount == vertices.size()){

		comm.allReduce(context, op, reduce, *(recvDatas[0]));

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

	    VAddr rootVAddr  = locateVertex(rootVertex);
	    Context context  = graphContext;

	    // Insert data of srcVertex to the end of the gather vector
	    gather.insert(gather.end(), sendData.begin(), sendData.end());

	    // Store recv pointer of rootVertex
	    if(srcVertex.id == rootVertex.id){
		rootRecvData = &recvData;
		peerHostsRootVertex = true;
	    }

	    if(nGatherCalls == hostedVertices.size()){
		std::vector<unsigned> recvCount;
		if(peerHostsRootVertex){
		    comm.gatherVar(rootVAddr, context, gather, *rootRecvData, recvCount);

		    // Reorder the received data, so that the data
		    // is in vertex id order. This operation is no
		    // sorting since the mapping is known before.
		    if(reorder){
			std::vector<RecvValueType> recvDataReordered(recvData.size());
			Cage::reorder(*rootRecvData, recvCount, recvDataReordered);
			std::copy(recvDataReordered.begin(), recvDataReordered.end(), rootRecvData->begin());

		    }
		
		}
		else {
		    comm.gatherVar(rootVAddr, context, gather, recvData, recvCount);
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
	    typedef typename T_Send::value_type SendValueType;
	    typedef typename T_Recv::value_type RecvValueType;

	    static std::vector<SendValueType> gather;
	    static std::vector<T_Recv*> recvDatas;
	    static unsigned nGatherCalls    = 0;
	    nGatherCalls++;
	    
	    VAddr srcVAddr  = locateVertex(srcVertex);
	    Context context = graphContext;
	    std::vector<Vertex> vertices = getHostedVertices(srcVAddr);

	    gather.insert(gather.end(), sendData.begin(), sendData.end());
	    recvDatas.push_back(&recvData);


	    if(nGatherCalls == hostedVertices.size()){
		std::vector<unsigned> recvCount;

		comm.allGatherVar(context, gather, *(recvDatas[0]), recvCount);

		// Reordering code
		if(reorder){
		    std::vector<unsigned> prefixsum(context.size(),0);

		    unsigned sum = 0;
		    for(unsigned count_i = 0; count_i < recvCount.size(); ++count_i){
			prefixsum[count_i] = sum;
			sum += recvCount[count_i];
		    }
		    
		    std::vector<RecvValueType> recvDataReordered(recvData.size());
		    for(unsigned vAddr = 0; vAddr < context.size(); vAddr++){
			std::vector<Vertex> hostedVertices = getHostedVertices(vAddr);
			unsigned nElementsPerVertex = recvCount.at(vAddr) / hostedVertices.size();

			unsigned hVertex_i=0;
			for(Vertex v: hostedVertices){

			    std::copy(recvDatas[0]->begin()+(prefixsum[vAddr] + (hVertex_i * nElementsPerVertex)),
				      recvDatas[0]->begin()+(prefixsum[vAddr] + (hVertex_i * nElementsPerVertex)) + (nElementsPerVertex),
				      recvDataReordered.begin()+(v.id*nElementsPerVertex));
			    hVertex_i++;

			}
			    
		    }

		}
		

		// if(reorder){
		//     std::vector<typename T_Recv::value_type> recvDataReordered(recvData.size());
		//     unsigned vAddr = 0;
		//     for(unsigned recv_i = 0; recv_i < recvData.size(); ){
		// 	std::vector<Vertex> hostedVertices = getHostedVertices(vAddr);
		// 	for(Vertex v: hostedVertices){
		// 	    recvDataReordered.at(v.id) = recvDatas[0]->data()[recv_i];
		// 	    recv_i++;
		// 	}
		// 	vAddr++;
		//     }
		//     for(unsigned i = 0; i < recvDataReordered.size(); ++i){
		// 	recvDatas[0]->data()[i] = recvDataReordered[i];
		//     }
		    
		// }

		// Distribute Received Data to Hosted Vertices
		//unsigned nElements = std::accumulate(recvCount.begin(), recvCount.end(), 0);
		for(unsigned i = 1; i < recvDatas.size(); ++i){
		    std::copy(recvDatas[0]->begin(), recvDatas[0]->end(), recvDatas[i]->begin());

		}
	    
		gather.clear();


	    }

	}

	void synchronize(){
	    comm.synchronize(graphContext);

	}
	/** @} */

    private:

	/**
	 * @brief Reorders data received from vertices into vertex id order.
	 *
	 */
	template <class T>
	void reorder(const std::vector<T> &data, const std::vector<unsigned> &recvCount, std::vector<T> &dataReordered){
	    std::vector<unsigned> prefixsum(graphContext.size(), 0);

	    utils::exclusivePrefixSum(recvCount.begin(), recvCount.end(), prefixsum.begin());

	    for(unsigned vAddr = 0; vAddr < graphContext.size(); vAddr++){
		const std::vector<Vertex> hostedVertices = getHostedVertices(vAddr);
		const unsigned nElementsPerVertex = recvCount.at(vAddr) / hostedVertices.size();

		for(unsigned hostVertex_i = 0; hostVertex_i < hostedVertices.size(); hostVertex_i++){

		    unsigned sourceOffset = prefixsum[vAddr] + (hostVertex_i * nElementsPerVertex);
		    unsigned targetOffset = hostedVertices[hostVertex_i].id * nElementsPerVertex;
		    
		    std::copy(data.begin() + sourceOffset,
			      data.begin() + sourceOffset + nElementsPerVertex,
			      dataReordered.begin() + targetOffset);
		}
	    }

	}
	
    };

} // namespace graybat


/**
 * @example gol.cc
 *
 * @brief Simple example that shows how to instantiate and use the Cage.
 *
 */
