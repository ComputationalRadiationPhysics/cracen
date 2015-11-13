namespace graybat {

  namespace pattern {

    typedef unsigned                                                        VertexID;
    typedef std::pair<VertexID, VertexID>                                   EdgeDescription;
    typedef std::pair<std::vector<VertexID>, std::vector<EdgeDescription> > GraphDescription;


    struct EdgeLess {

      const unsigned verticesCount;

	EdgeLess(const unsigned verticesCount) :
	    verticesCount(verticesCount){

      }
      
      GraphDescription operator()(){
	std::vector<VertexID> vertices(verticesCount);
	assert(vertices.size() == verticesCount);
	std::vector<EdgeDescription> edges;
	return std::make_pair(vertices,edges);
      }

    };
      
  } /* pattern */

} /* graybat */
