import Vue from 'vue'
import VueRouter from 'vue-router'
import axios from 'axios';
import VueTimeago from 'vue-timeago'

import App from './App.vue'
import TopHeader from './components/TopHeader.vue'
import EventBus from './components/EventBus'

Vue.use(VueRouter)

Vue.use(VueTimeago, {
	name: 'timeago', // component name, `timeago` by default
	locale: 'zh-CN',
	locales: {
		// you will need json-loader in webpack 1
		'zh-CN': require('vue-timeago/locales/zh-CN.json')
	}
})

// 注册组件
Vue.component(TopHeader.name, TopHeader)

const routes = [{
		path: '/articles',
		component: resolve => (require(["./components/router/Article.vue"], resolve)),
		children: [{
			path: 'detail/:id',
			name: 'article-detail',
			component: resolve => (require(["./components/ArticleDetailTab.vue"], resolve)),
		}]
	},
	{
		path: '/topics',
		component: resolve => (require(["./components/router/Topic.vue"], resolve)),
		children: [{
			path: 'detail/:id',
			name: 'topic-detail',
			component: resolve => (require(["./components/TopicDetailTab.vue"], resolve)),
		}]
	},
	{
		path: '/resources',
		component: resolve => (require(["./components/router/Resource.vue"], resolve)),
		children: [{
			path: 'detail/:id',
			name: 'resource-detail',
			component: resolve => (require(["./components/ResourceDetailTab.vue"], resolve)),
		}]
	},
	{
		path: '/projects',
		component: resolve => (require(["./components/router/Project.vue"], resolve)),
		children: [{
			path: 'detail/:id',
			name: 'project-detail',
			component: resolve => (require(["./components/ProjectDetailTab.vue"], resolve)),
		}]
	},
	{
		path: '/user',
		component: resolve => (require(["./components/router/User.vue"], resolve)),
		children: [{
				path: 'login',
				component: resolve => (require(["./components/Login.vue"], resolve)),
			},
			{
				path: 'info',
				component: resolve => (require(["./components/UserInfo.vue"], resolve)),
			},
			{
				path: 'about',
				component: resolve => (require(["./components/AboutTab.vue"], resolve)),
			}
		]
	}
];

const router = new VueRouter({
	routes
})

/* 全局拦截器 */
// request 拦截器
axios.interceptors.request.use(function (config) {
	var method = config.method.toUpperCase();
	EventBus.$emit('showLoading', method == 'GET' ? '加载中' : '提交中'); // 展示loading
	return config;
}, function (error) {
	// Do something with request error
	return Promise.reject(error);
});
// response 拦截器
axios.interceptors.response.use(
	function (resp) {
		EventBus.$emit('hideLoading'); // 隐藏loading
		var data = resp.data;
		var code = data.code;
		if (typeof code !== "undefined" && code != 0) {
			EventBus.$emit('alert', {
				message: data.msg || '未知错误',
				callback: () => {
					if (code == 600) {
						router.push('/user/login');
					}
				}
			});
			resp.abort = true;
		}
		return resp;
	},
	function (err) {
		EventBus.$emit('hideLoading'); // 隐藏loading
		EventBus.$emit('alert', {
			message: '服务器异常'
		});
		return Promise.reject(err);
	}
);

/**
 * 给原生Promise增加finally方法, 不论是resolve还是reject都会执行
 * @param callback
 * @returns {Promise.<TResult>}
 */
Promise.prototype.finally = function (callback) {
	let P = this.constructor;
	return this.then(
		value => P.resolve(callback()).then(() => value),
		reason => P.resolve(callback()).then(() => {
			throw reason
		})
	);
};

new Vue({
	el: '#app',
	router,
	render: h => h(App)
})

// 默认跳转到第一个菜单
if (router.currentRoute.path == '/') {
	router.push(routes[0].path);
}

// 禁止微信的傻逼下拉功能，和👴写的下拉效果冲突了。
// 但👴只修好了首页的下拉冲突，别的页面还是会收到傻逼微信的傻逼下拉效果的傻逼影响，待优化。
document.body.addEventListener('touchmove', function (e) {
	e.preventDefault();
}, {
	passive: false
});