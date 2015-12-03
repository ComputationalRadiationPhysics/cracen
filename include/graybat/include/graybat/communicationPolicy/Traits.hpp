#pragma once

namespace graybat {
    
    namespace communicationPolicy {

        namespace traits {

        template <typename T_CommunicationPolicy>
        struct ContextType;

        template <typename T_CommunicationPolicy>
        struct EventType;

        template <typename T_CommunicationPolicy>
        struct ConfigType;
            
        } // traits

        template <typename T_CommunicationPolicy>        
        using VAddr = unsigned;

        template <typename T_CommunicationPolicy>
        using Tag = unsigned;

        template <typename T_CommunicationPolicy>        
        using ContextID = unsigned;

        template <typename T_CommunicationPolicy>        
        using MsgType = unsigned;

        template <typename T_CommunicationPolicy>        
        using MsgID = unsigned;

        template <typename T_CommunicationPolicy>
        using Context = typename traits::ContextType<T_CommunicationPolicy>::type;

        template <typename T_CommunicationPolicy>
        using Event = typename traits::EventType<T_CommunicationPolicy>::type;

        template <typename T_CommunicationPolicy>
        using Config = typename traits::ConfigType<T_CommunicationPolicy>::type;        
        
    } // namespace communicationPolicy
    
} // namespace graybat
