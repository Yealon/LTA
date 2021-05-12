<template>
  <transition name="aside-tab">
    <aside class="aside-tab">
      <s-header title="法律案由预测" back="1"></s-header>
      <div class="article-wrap">
        <h4>AI能力key：legal_cause</h4>
        <ul class="article-footer-meta">
          <li>【算法输入】 一段案件描述：3000字符以内，只支持中文。</li>
          <li>【算法输出】 几个案由分类：一组关键词。</li>
        </ul>
        <div class="form">
          <form role="form">
            <div class="form-group">
              <label for="name">案件描述：</label>
              <textarea
                class="form-control"
                rows="10"
                cols="5"
                placeholder="输入内容长度需介于20~3000字"
                v-model="description"
              ></textarea>
            </div>
            <div class="form-group" style="float: right">
              <div class="col-sm-10">
                <button
                  type="submit"
                  class="btn btn-success"
                  v-on:click="analyse()"
                >
                  分析
                </button>
                <button
                  type="submit"
                  class="btn btn-danger"
                  v-on:click="clear()"
                >
                  清空
                </button>
              </div>
            </div>
            <div class="answer" style="margin-top: 70px" v-show="showRs">
              <div class="form-group">
                <label for="name">预测结果：</label>
                <div class="article-footer-meta">
                  <span class="label label-warning">分类1</span>
                  <span class="label label-warning">分类2</span>
                  <span class="label label-warning">分类3</span>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>
      <div class="loading-tip" v-show="!article.id" v-cloak>
        该文章不存在或被删除
      </div>
    </aside>
  </transition>
</template>

<script>
import NAProxy from "./NAProxy";
import { getToken } from "./TokenUtils";
import { URL_ARTICLE_DETAIL } from "./Config";
import axios from "axios";
import EventBus from "../components/EventBus";
import { URL_AI_LEGAL } from "./Config";

export default {
  name: "article-detail-tab",
  data: function () {
    return {
      description: "",
      showRs: false,
      article_id: 0,
      article: { id: 1 },
      rs: ["", "", ""],
    };
  },
  mounted: function () {
    this.article_id = this.$route.params.id;
    this.getData();
  },
  methods: {
    analyse: function () {
      if (this.description.length < 20 || this.description.length > 3000) {
        this.showRs = false;
        EventBus.$emit("alert", {
          message: "案件描述需要介于20~3000字！",
        });
      } else {
        axios.post(URL_AI_LEGAL, this.description).then((resp) => {
          if (resp.abort) return;
          this.showRs = true; // 防止卡顿
          this.rs = rs.legal_tag;
        });
      }
    },
    clear: function () {
      this.showRs = false;
      this.description = "";
    },
    getData: function () {
      return;
      getToken().then(
        (token) => {
          axios
            .get(URL_ARTICLE_DETAIL, {
              params: {
                id: this.article_id,
                token: token,
              },
            })
            .then((resp) => {
              if (resp.abort) return;

              var data = resp.data.data;
              this.article = data.article;
            });
        },
        (err) => {
          this.alert(err);
        }
      );
    },
  },
};
</script>

<style lang="sass" scoped>
@import "../sass/DetailTab.scss"
</style>